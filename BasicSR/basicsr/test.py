import logging
import torch
from os import path as osp

# 确保注册你自定义的数据集
from lightdiff.data import lol_dataset  # noqa: F401

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def test_pipeline(root_path):
    # 解析配置
    opt, _ = parse_options(root_path, is_train=False)

    # Windows/单机：禁用 distributed
    if 'dist' in opt:
        opt['dist'] = False

    torch.backends.cudnn.benchmark = True

    # 日志与实验目录
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # dataloader（支持 datasets:test 或 datasets:val）
    test_loaders = []
    ds_dict = opt.get('datasets', {})
    if 'test' in ds_dict:
        data_opt = ds_dict['test']
        test_set = build_dataset(data_opt)
        test_loader = build_dataloader(
            test_set, data_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test items in {data_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)
    if 'val' in ds_dict:
        data_opt = ds_dict['val']
        test_set = build_dataset(data_opt)
        test_loader = build_dataloader(
            test_set, data_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of val items in {data_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    if not test_loaders:
        raise RuntimeError("No dataset found under opt['datasets']. Please define 'test' or 'val'.")

    # 构建模型并切 eval
    model = build_model(opt)
    for sub in ['ddpm', 'temporal_se', 'unet', 'global_corrector']:
        if hasattr(model, sub) and getattr(model, sub) is not None:
            getattr(model, sub).eval()

    # 统一在 test/val loader 上跑 validation（保存三联图/视频并计算 metrics）
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        # current_iter 用 0 即可；会写入三联图名中
        model.validation(test_loader, current_iter=0, tb_logger=None, save_img=opt['val'].get('save_img', True))


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
