import datetime
import logging
import math
import time
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options


def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
    return resume_state


def train_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    # optional early stopping settings
    early_stop_opt = opt['train'].get('early_stop') if opt.get('train') else None
    early_stop_state = None
    if early_stop_opt:
        if not val_loaders or opt.get('val') is None or opt['val'].get('metrics') is None:
            logger.warning('`train.early_stop` is set but validation metrics are unavailable; early stopping is disabled.')
            early_stop_opt = None
        else:
            metric_name = early_stop_opt.get('metric')
            if not metric_name:
                logger.warning('`train.early_stop.metric` is required; early stopping is disabled.')
                early_stop_opt = None
            else:
                metric_cfg = opt['val']['metrics'].get(metric_name)
                if metric_cfg is None:
                    logger.warning('Early stopping metric `%s` is not defined in `val.metrics`; disabling early stopping.',
                                   metric_name)
                    early_stop_opt = None
                else:
                    better = early_stop_opt.get('better', metric_cfg.get('better', 'higher'))
                    if better not in ('higher', 'lower'):
                        logger.warning('Invalid `train.early_stop.better` value `%s`; falling back to `higher`.', better)
                        better = 'higher'
                    patience = int(early_stop_opt.get('patience', 5))
                    min_delta = float(early_stop_opt.get('min_delta', 0.0))
                    dataset_name = early_stop_opt.get('dataset')
                    if dataset_name is None and len(val_loaders) > 1:
                        first_dataset_opt = getattr(val_loaders[0].dataset, 'opt', {})
                        if isinstance(first_dataset_opt, dict):
                            dataset_name = first_dataset_opt.get('name', 'val')
                        else:
                            dataset_name = 'val'
                        logger.warning('Multiple validation datasets detected but `train.early_stop.dataset` is not set; '
                                       'defaulting to `%s`.', dataset_name)
                    best_val = float('-inf') if better == 'higher' else float('inf')
                    early_stop_state = dict(metric=metric_name,
                                            better=better,
                                            patience=patience,
                                            min_delta=min_delta,
                                            dataset=dataset_name,
                                            best_val=best_val,
                                            best_iter=-1,
                                            num_bad_epochs=0,
                                            triggered=False)

    # create model
    model = build_model(opt)
    if resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    stop_training = False
    for epoch in range(start_epoch, total_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_timer.record()
            if current_iter == 1:
                # reset start time in msg_logger for more accurate eta_time
                # not work in resume mode
                msg_logger.reset_start_time()
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                if len(val_loaders) > 1:
                    logger.warning('Multiple validation datasets are *only* supported by SRModel.')
                for val_loader in val_loaders:
                    dataset_name = val_loader.dataset.opt.get('name', 'val') if hasattr(val_loader.dataset, 'opt') else None
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
                    if early_stop_state and not early_stop_state['triggered']:
                        target_dataset = early_stop_state['dataset'] or dataset_name
                        if target_dataset == dataset_name or early_stop_state['dataset'] is None:
                            metric_results = getattr(model, 'metric_results', None)
                            if metric_results is None or early_stop_state['metric'] not in metric_results:
                                logger.warning('Metric results for `%s` are unavailable; disabling early stopping.',
                                               early_stop_state['metric'])
                                early_stop_state = None
                            else:
                                current_val = float(metric_results[early_stop_state['metric']])
                                if early_stop_state['better'] == 'higher':
                                    improved = current_val > (early_stop_state['best_val'] + early_stop_state['min_delta'])
                                else:
                                    improved = current_val < (early_stop_state['best_val'] - early_stop_state['min_delta'])
                                if improved:
                                    early_stop_state['best_val'] = current_val
                                    early_stop_state['best_iter'] = current_iter
                                    early_stop_state['num_bad_epochs'] = 0
                                else:
                                    early_stop_state['num_bad_epochs'] += 1
                                    if early_stop_state['num_bad_epochs'] >= early_stop_state['patience']:
                                        early_stop_state['triggered'] = True
                                        stop_training = True
                                        logger.info(
                                            'Early stopping triggered at iter %d: %s/%s=%.6f (best %.6f @ iter %d).',
                                            current_iter,
                                            dataset_name or 'val',
                                            early_stop_state['metric'],
                                            current_val,
                                            early_stop_state['best_val'],
                                            early_stop_state['best_iter'])
                                        break
                
                if stop_training:
                    break

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()
        if stop_training:
            break
        # end of iter

    # end of epoch
        if stop_training:
            break

    if early_stop_state and early_stop_state.get('triggered'):
        logger.info('Training stopped early after reaching patience limit.')

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
