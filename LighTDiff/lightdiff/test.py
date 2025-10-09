# =============================== 单文件可跑：test_pipeline & TemporalSE ===============================
import logging
import torch
from os import path as osp
import lightdiff.data.lol_dataset  # 注册你的数据集

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def test_pipeline(root_path):
    # parse options
    opt, _ = parse_options(root_path, is_train=False)
    torch.backends.cudnn.benchmark = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed']
        )
        logger.info(f"Number of test images/videos in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    # 将内部网络切到 eval
    if hasattr(model, 'ddpm') and model.ddpm is not None:
        model.ddpm.eval()
    if getattr(model, 'temporal_se', None) is not None:
        model.temporal_se.eval()
    if hasattr(model, 'unet') and model.unet is not None:
        model.unet.eval()
    if hasattr(model, 'global_corrector') and model.global_corrector is not None:
        model.global_corrector.eval()

    # run validation
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, current_iter=0, tb_logger=None, save_img=opt['val']['save_img'])


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
