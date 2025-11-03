import json
import os
import os.path as osp
import time
from collections import OrderedDict

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torchvision.ops import roi_align  # 历史依赖，保留
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.archs.arch_util import flow_warp
from basicsr.losses import build_loss
from basicsr.losses.losses import r1_penalty
from basicsr.metrics import calculate_metric
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY

from scripts.utils import pad_tensor_back
from thop import profile  # 历史依赖，保留

cv2.setNumThreads(1)


@MODEL_REGISTRY.register()
class LighTDiff(BaseModel):

    # -------------------- 视频写入器（OpenCV 优先，失败回退 imageio-ffmpeg） --------------------
    class _VideoSink:
        def __init__(self, results_dir: str):
            self.results_dir = results_dir
            self.cv2_writers = {}      # key -> cv2.VideoWriter
            self.imageio_writers = {}  # key -> imageio Writer
            self.meta = {}             # key -> (w, h, fps, ext)
            self.sizes = {}            # key -> (w, h)

        def _open_cv2(self, path_mp4, w, h, fps):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vw = cv2.VideoWriter(path_mp4, fourcc, fps, (w, h))
            return vw if vw.isOpened() else None

        def _open_imageio(self, path_mp4, fps):
            # 需要: pip install imageio imageio-ffmpeg
            return imageio.get_writer(path_mp4, fps=float(fps), codec='libx264', quality=8)

        def get(self, key: str, w: int, h: int, fps: float):
            if key in self.cv2_writers or key in self.imageio_writers:
                return  # 已初始化
            os.makedirs(self.results_dir, exist_ok=True)
            mp4_path = osp.join(self.results_dir, f'{key}.mp4')

            # 先试 cv2
            vw = self._open_cv2(mp4_path, w, h, fps)
            if vw is not None:
                self.cv2_writers[key] = vw
                self.meta[key] = (w, h, fps, '.mp4')
                self.sizes[key] = (w, h)
                get_root_logger().info(f"[VideoSink] use OpenCV writer: {mp4_path}, fps={fps}, size=({w}x{h})")
                return

            # 回退 imageio
            iw = self._open_imageio(mp4_path, fps)
            self.imageio_writers[key] = iw
            self.meta[key] = (w, h, fps, '.mp4')
            self.sizes[key] = (w, h)
            get_root_logger().info(f"[VideoSink] fallback to imageio writer: {mp4_path}, fps={fps}, size=({w}x{h})")

        def write(self, key: str, frame_rgb_uint8: np.ndarray):
            """
            约定: 传入帧是 RGB/uint8/HxWx3/连续内存
            - 对于当前 OpenCV 构建，直接写入 RGB（不做任何通道翻转）
            - imageio 也直接写 RGB
            """
            if key in self.cv2_writers:
                self.cv2_writers[key].write(frame_rgb_uint8)       # 直接写 RGB
            elif key in self.imageio_writers:
                self.imageio_writers[key].append_data(frame_rgb_uint8)  # 仍是 RGB
            else:
                raise RuntimeError(f'VideoSink for key={key} not initialized')


        def close_all(self):
            for k, vw in self.cv2_writers.items():
                try:
                    vw.release()
                except Exception:
                    pass
            for k, iw in self.imageio_writers.items():
                try:
                    iw.close()
                except Exception:
                    pass
            self.cv2_writers.clear()
            self.imageio_writers.clear()
            self.meta.clear()
            self.sizes.clear()


    # --------------------------------------------------------------------------------------------

    def __init__(self, opt):
        super(LighTDiff, self).__init__(opt)

        # === Temporal SE 开关（轻量时间加权融合） ===
        self.use_temporal_se = opt.get('train', {}).get('use_temporal_se', False)
        if self.use_temporal_se:
            from lightdiff.models.temporal_se import TemporalSE
            # clip_len 从 datasets.train / datasets.val 里任选；默认 5
            t_len = (
                opt.get('datasets', {}).get('train', {}).get('clip_len', None)
                or opt.get('datasets', {}).get('val', {}).get('clip_len', 5)
            )
            self.temporal_se = TemporalSE(channels=3, t=t_len)
            self.temporal_se = self.model_to_device(self.temporal_se)
        else:
            self.temporal_se = None

        # 融合策略（默认: 有 TemporalSE 时走加权融合，否则取中心帧）
        self.temporal_fusion_mode_train = opt.get('train', {}).get('temporal_fusion', None)
        self.temporal_fusion_mode_eval = opt.get('val', {}).get('temporal_fusion', None)
        if self.temporal_fusion_mode_eval is None:
            self.temporal_fusion_mode_eval = self.temporal_fusion_mode_train
        self._temporal_fusion_logged = False

        # === 额外的时序对齐骨干（BasicVSRNet / EDVR 等） ===
        self.temporal_backbone = None
        self.temporal_backbone_center_idx = None
        temporal_backbone_opt = opt.get('network_temporal_backbone', None)
        if temporal_backbone_opt:
            self.temporal_backbone = build_network(temporal_backbone_opt)
            self.temporal_backbone = self.model_to_device(self.temporal_backbone)
            self.temporal_backbone_center_idx = temporal_backbone_opt.get('center_frame_idx', None)
            self.print_network(self.temporal_backbone)

        # === 构建网络 ===
        self.unet = build_network(opt['network_unet'])
        self.unet = self.model_to_device(self.unet)
        opt['network_ddpm']['denoise_fn'] = self.unet

        self.global_corrector = build_network(opt['network_global_corrector'])
        self.global_corrector = self.model_to_device(self.global_corrector)
        opt['network_ddpm']['network_global_corrector'] = self.global_corrector

        self.ddpm = build_network(opt['network_ddpm'])
        self.ddpm = self.model_to_device(self.ddpm)
        if isinstance(self.ddpm, (DataParallel, DistributedDataParallel)):
            self.bare_model = self.ddpm.module
        else:
            self.bare_model = self.ddpm

        self.bare_model.set_new_noise_schedule(
            schedule_opt=opt['ddpm_schedule'], device=self.device)
        self.bare_model.set_loss(device=self.device)
        self.print_network(self.ddpm)

        self.optimizer_temporal = None

        # 载入预训练
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.ddpm, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key)

        # LPIPS（验证与训练共用）
        self.lpips = None
        self.lpips_bare_model = None
        metrics_cfg_check = (self.opt.get('val', {}) or {}).get('metrics') or self.opt.get('metrics')
        train_lpips_w = float(self.opt.get('train', {}).get('l_g_lpips_w', 0.0) or 0.0)
        if metrics_cfg_check and any('lpips' in m for m in metrics_cfg_check.keys()):
            import lpips
            self.lpips = lpips.LPIPS(net='alex')
            self.lpips = self.model_to_device(self.lpips)
            self.lpips_bare_model = self.lpips.module if isinstance(self.lpips, (DataParallel, DistributedDataParallel)) else self.lpips
        elif train_lpips_w > 0:
            import lpips
            self.lpips = lpips.LPIPS(net=self.opt.get('train', {}).get('lpips_net', 'alex'))
            self.lpips = self.model_to_device(self.lpips)
            self.lpips_bare_model = self.lpips.module if isinstance(self.lpips, (DataParallel, DistributedDataParallel)) else self.lpips

        self.lpips_loss_weight = train_lpips_w
        if self.lpips is not None:
            self.lpips.eval()
            for p in self.lpips.parameters():
                p.requires_grad_(False)
        self.luma_regularizer_weight = float(self.opt.get('train', {}).get('l_g_brightness_w', 0.0) or 0.0)
        self.temporal_consistency_weight = float(self.opt.get('train', {}).get('l_g_temporal_w', 0.0) or 0.0)

        self.temporal_sequence = None
        self.hr_sequence = None

        # === 训练过程统计 & 可视化输出 ===
        self.rank = int(opt.get('rank', 0) or 0)
        analytics_dir = osp.join(opt['path'].get('experiments_root', '.'), 'analytics')
        if self.rank == 0:
            os.makedirs(analytics_dir, exist_ok=True)
        self.analytics_dir = analytics_dir
        self.training_history_path = osp.join(analytics_dir, 'training_history.jsonl')
        self.validation_history_path = osp.join(analytics_dir, 'validation_history.jsonl')
        logger_cfg = opt.get('logger', {}) or {}
        self.analytics_log_interval = int(logger_cfg.get('print_freq', 100) or 1)

        # test 阶段补默认 results 路径
        if 'path' in self.opt and 'results' not in self.opt['path']:
            self.opt['path']['results'] = osp.join(
                self.opt['path'].get('experiments_root', '.'), 'results'
            )

        if self.is_train:
            self.init_training_settings()

        # 亮度先验缓存
        self.lr_luma_context = None
        self.lr_luma_center = None
        self.lr_luma_window = None

    @staticmethod
    def _rgb_to_luma_mean(tensor):
        """将 [-1,1] 归一化 RGB 张量转换为均值亮度（Y 通道）。"""
        if tensor.dim() != 4:
            raise ValueError(f'_rgb_to_luma_mean expects 4D tensor, got shape={tensor.shape}')
        coeffs = tensor.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
        tensor_01 = (tensor + 1.0) * 0.5
        luma = (tensor_01 * coeffs).sum(dim=1)
        return luma.mean(dim=[1, 2])

    @staticmethod
    def _seq_luma_stats(seq, center_idx):
        """
        计算输入序列（B,T,3,H,W）的中心亮度与时间窗口平均亮度。
        返回 (center_mean, window_mean) 均为 [B]。
        """
        if seq.dim() != 5:
            raise ValueError(f'_seq_luma_stats expects 5D tensor, got shape={seq.shape}')
        coeffs = seq.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 1, 3, 1, 1)
        seq_01 = (seq + 1.0) * 0.5
        luma = (seq_01 * coeffs).sum(dim=2)
        # sum over spatial dims (H, W)
        luma_mean = luma.mean(dim=[2, 3])  # [B,T]
        center = luma_mean[:, center_idx]
        window = luma_mean.mean(dim=1)
        return center, window

    # ====== 训练初始化 ====== #
    def init_training_settings(self):
        self.ddpm.train()
        if self.temporal_backbone is not None:
            self.temporal_backbone.train()
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        net_g_reg_ratio = 1
        normal_params = []
        logger = get_root_logger()
        for name, param in self.ddpm.named_parameters():
            if self.opt['train'].get('frozen_denoise', False) and 'denoise' in name:
                logger.info(f'frozen {name}')
                continue
            normal_params.append(param)
        optim_g_opt = train_opt['optim_g'].copy()
        optim_params_g = [{'params': normal_params, 'lr': optim_g_opt['lr']}]
        optim_type = optim_g_opt.pop('type')
        lr = float(optim_g_opt.pop('lr') * net_g_reg_ratio)
        betas = (0 ** net_g_reg_ratio, 0.99 ** net_g_reg_ratio)
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, lr, betas=betas, **optim_g_opt)
        self.optimizers.append(self.optimizer_g)

        if self.temporal_backbone is not None and train_opt.get('optim_temporal'):
            temporal_opt = train_opt['optim_temporal'].copy()
            temporal_type = temporal_opt.pop('type')
            temporal_lr = temporal_opt.pop('lr')
            temporal_params = [p for p in self.temporal_backbone.parameters() if p.requires_grad]
            if temporal_params:
                optim_params_t = [{'params': temporal_params, 'lr': temporal_lr}]
                self.optimizer_temporal = self.get_optimizer(
                    temporal_type, optim_params_t, temporal_lr, **temporal_opt)
                self.optimizers.append(self.optimizer_temporal)

    # ====== 数据喂入（兼容视频/图像） ====== #
    def feed_data(self, data):
        self.temporal_reg_pred = None
        self.temporal_reg_target = None
        self.lr_luma_context = None
        self.lr_luma_center = None
        self.lr_luma_window = None
        self.temporal_sequence = None
        self.hr_sequence = None
        if 'LR_seq' in data and 'HR_mid' in data:
            LR_seq = data['LR_seq'].to(self.device)
            HR_mid = data['HR_mid'].to(self.device)
            if LR_seq.dim() == 4:  # [T,3,H,W]
                LR_seq = LR_seq.unsqueeze(0)
            if HR_mid.dim() == 3:  # [3,H,W]
                HR_mid = HR_mid.unsqueeze(0)
            HR_seq = data.get('HR_seq', None)
            if HR_seq is not None:
                HR_seq = HR_seq.to(self.device)
                if HR_seq.dim() == 4:
                    HR_seq = HR_seq.unsqueeze(0)

            center = LR_seq.shape[1] // 2
            if self.temporal_backbone_center_idx is not None:
                center = min(max(int(self.temporal_backbone_center_idx), 0), LR_seq.shape[1] - 1)
            center_frame = LR_seq[:, center, ...]
            processed_seq = LR_seq
            with torch.no_grad():
                center_luma, window_luma = self._seq_luma_stats(LR_seq, center)
                self.lr_luma_center = center_luma
                self.lr_luma_window = window_luma
                self.lr_luma_context = window_luma
            if self.temporal_backbone is not None:
                processed_seq = self.temporal_backbone(processed_seq)
                if processed_seq.dim() == 4:
                    processed_seq = processed_seq.unsqueeze(1)

            self.temporal_sequence = processed_seq
            self.hr_sequence = HR_seq

            fusion_mode = self._resolve_temporal_fusion_mode()
            if not self._temporal_fusion_logged:
                backbone_state = 'on' if self.temporal_backbone is not None else 'off'
                se_state = 'on' if self.temporal_se is not None else 'off'
                get_root_logger().info(
                    f"[TemporalFusion] mode={fusion_mode}, backbone={backbone_state}, temporal_se={se_state}"
                )
                self._temporal_fusion_logged = True

            if fusion_mode == 'se':
                if self.temporal_se is None:
                    get_root_logger().warning(
                        '[TemporalFusion] temporal_fusion="se" 但 TemporalSE 未启用，退化为 center 模式'
                    )
                    self.LR = processed_seq[:, center, ...]
                else:
                    self.LR = self.temporal_se(processed_seq)  # [B,3,H,W]
            elif fusion_mode in {'aligned_center', 'center'}:
                if fusion_mode == 'aligned_center' and self.temporal_backbone is None:
                    get_root_logger().warning(
                        '[TemporalFusion] temporal_fusion="aligned_center" 但未构建时序骨干，退化为 center 模式'
                    )
                self.LR = processed_seq[:, center, ...]
            else:
                raise ValueError(f'Unsupported temporal fusion mode: {fusion_mode}')

            self.temporal_reg_pred = processed_seq[:, center, ...]
            self.temporal_reg_target = center_frame
            self.lq_vis = center_frame.clone()
            self.HR = HR_mid
        else:
            self.LR = data['LR'].to(self.device)
            self.HR = data['HR'].to(self.device)
            self.lq_vis = self.LR[:, :3, ...].clone()
            with torch.no_grad():
                center_luma = self._rgb_to_luma_mean(self.lq_vis)
                self.lr_luma_center = center_luma
                self.lr_luma_window = center_luma
                self.lr_luma_context = center_luma

        # 可选 padding
        for k in ['pad_left', 'pad_right', 'pad_top', 'pad_bottom']:
            if k in data:
                setattr(self, k, data[k].to(self.device))

    def _resolve_temporal_fusion_mode(self):
        default_mode = 'se' if self.temporal_se is not None else 'center'
        # 训练配置优先；若未指定则尝试验证/测试项
        if self.opt.get('is_train', False):
            mode = self.temporal_fusion_mode_train
            if mode is None:
                mode = self.temporal_fusion_mode_eval
        else:
            mode = self.temporal_fusion_mode_eval
        if mode is None:
            mode = default_mode
        return mode

    def _run_ddpm(self, x_hr, x_lr):
        train_opt = self.opt['train']
        return self.ddpm(
            x_hr,
            x_lr,
            train_type=train_opt.get('train_type', None),
            different_t_in_one_batch=train_opt.get('different_t_in_one_batch', None),
            t_sample_type=train_opt.get('t_sample_type', None),
            pred_type=train_opt.get('pred_type', None),
            clip_noise=train_opt.get('clip_noise', None),
            color_shift=train_opt.get('color_shift', None),
            color_shift_with_schedule=train_opt.get('color_shift_with_schedule', None),
            t_range=train_opt.get('t_range', None),
            cs_on_shift=train_opt.get('cs_on_shift', None),
            cs_shift_range=train_opt.get('cs_shift_range', None),
            t_border=train_opt.get('t_border', None),
            down_uniform=train_opt.get('down_uniform', False),
            down_hw_split=train_opt.get('down_hw_split', False),
            pad_after_crop=train_opt.get('pad_after_crop', False),
            input_mode=train_opt.get('input_mode', None),
            crop_size=train_opt.get('crop_size', None),
            divide=train_opt.get('divide', None),
            frozen_denoise=train_opt.get('frozen_denoise', None),
            cs_independent=train_opt.get('cs_independent', None),
            shift_x_recon_detach=train_opt.get('shift_x_recon_detach', None)
        )

    def _get_temporal_backbone_core(self):
        backbone = self.temporal_backbone
        if isinstance(backbone, (DataParallel, DistributedDataParallel)):
            backbone = backbone.module
        return backbone

    def _append_jsonl(self, path, record):
        if self.rank != 0:
            return
        if path is None:
            return
        try:
            with open(path, 'a', encoding='utf-8') as f:
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')
        except Exception as exc:
            get_root_logger().warning(f'Failed to write analytics record to {path}: {exc}')

    def _record_training_history(self, current_iter, log_dict):
        if self.rank != 0:
            return
        interval = max(1, self.analytics_log_interval)
        if current_iter % interval != 0:
            return

        record = {
            'iter': int(current_iter),
            'timestamp': time.time(),
        }
        try:
            lrs = self.get_current_learning_rate()
        except Exception:
            lrs = []
        for idx, lr in enumerate(lrs):
            record[f'lr_{idx}'] = float(lr)

        for key, value in (log_dict or {}).items():
            if value is None:
                continue
            if torch.is_tensor(value):
                value = value.item()
            record[key] = float(value)

        self._append_jsonl(self.training_history_path, record)

    def _record_validation_history(self, current_iter, dataset_name, metrics):
        if self.rank != 0:
            return
        record = {
            'iter': int(current_iter),
            'dataset': dataset_name,
            'timestamp': time.time(),
        }
        for key, value in (metrics or {}).items():
            if value is None:
                continue
            if torch.is_tensor(value):
                value = value.item()
            record[key] = float(value)
        self._append_jsonl(self.validation_history_path, record)

    def _compute_temporal_consistency(self):
        if self.temporal_consistency_weight <= 0:
            return None
        if self.temporal_sequence is None or self.hr_sequence is None:
            return None

        seq_lr = self.temporal_sequence
        seq_hr = self.hr_sequence
        if seq_lr is None or seq_hr is None:
            return None
        if seq_lr.dim() != 5 or seq_hr.dim() != 5:
            return None

        b, t, c, h, w = seq_lr.shape
        if t < 2:
            return None

        seq_lr_flat = seq_lr.reshape(b * t, c, h, w)
        seq_hr_flat = seq_hr.reshape(b * t, c, h, w)
        _, _, seq_recon, _, _, _ = self._run_ddpm(seq_hr_flat, seq_lr_flat)

        bt, c_rec, h_rec, w_rec = seq_recon.shape
        if bt != b * t:
            raise RuntimeError(
                f"Temporal consistency expects {b * t} frames but got {bt} from DDPM"
            )
        seq_recon = seq_recon.view(b, t, c_rec, h_rec, w_rec)

        backbone = self._get_temporal_backbone_core()
        flows_forward = flows_backward = None
        if backbone is not None and hasattr(backbone, 'get_flow'):
            flows_forward, flows_backward = backbone.get_flow(seq_recon)

        temporal_terms = []
        for i in range(t - 1):
            cur = seq_recon[:, i, ...]
            nxt = seq_recon[:, i + 1, ...]
            if flows_forward is not None and flows_backward is not None:
                flow_fw = flows_forward[:, i, ...].permute(0, 2, 3, 1)
                flow_bw = flows_backward[:, i, ...].permute(0, 2, 3, 1)
                nxt_to_cur = flow_warp(nxt, flow_fw)
                cur_to_nxt = flow_warp(cur, flow_bw)
                diff_forward = torch.abs(nxt_to_cur - cur).mean(dim=[1, 2, 3])
                diff_backward = torch.abs(cur_to_nxt - nxt).mean(dim=[1, 2, 3])
                pair_loss = 0.5 * (diff_forward + diff_backward)
            else:
                pair_loss = torch.abs(cur - nxt).mean(dim=[1, 2, 3])
            temporal_terms.append(pair_loss)

        if not temporal_terms:
            return None

        stacked = torch.stack(temporal_terms, dim=0)  # [T-1, B]
        return stacked.mean()

    # ====== 训练一步 ====== #
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        if self.optimizer_temporal is not None:
            self.optimizer_temporal.zero_grad()
        pred_noise, noise, x_recon_cs, x_start, t, color_scale = self._run_ddpm(self.HR, self.LR)

        # 训练可视
        if self.opt['train'].get('vis_train', False) and \
           current_iter <= self.opt['train'].get('vis_num', 100) and \
           self.opt['rank'] == 0:
            save_img_path = osp.join(self.opt['path']['visualization'], 'train',
                                     f'{current_iter}_noise_level_{self.bare_model.t}.png')
            x_recon_print = tensor2img(self.bare_model.x_recon, min_max=(-1, 1))
            noise_print = tensor2img(self.bare_model.noise, min_max=(-1, 1))
            pred_noise_print = tensor2img(self.bare_model.pred_noise, min_max=(-1, 1))
            x_start_print = tensor2img(self.bare_model.x_start, min_max=(-1, 1))
            x_noisy_print = tensor2img(self.bare_model.x_noisy, min_max=(-1, 1))
            img_print = np.concatenate(
                [x_start_print, noise_print, x_noisy_print, pred_noise_print, x_recon_print], axis=1)
            imwrite(img_print, save_img_path)

        # 损失
        l_g_total = 0
        loss_dict = OrderedDict()
        l_g_x0 = F.smooth_l1_loss(x_recon_cs, x_start, beta=0.5) * self.opt['train'].get('l_g_x0_w', 1.0)
        if self.opt['train'].get('gamma_limit_train', None) and \
           color_scale <= self.opt['train'].get('gamma_limit_train', None):
            l_g_x0 = l_g_x0 * 0.5
        loss_dict['l_g_x0'] = l_g_x0
        l_g_total += l_g_x0

        if not self.opt['train'].get('frozen_denoise', False):
            l_g_noise = F.smooth_l1_loss(pred_noise, noise)
            loss_dict['l_g_noise'] = l_g_noise
            l_g_total += l_g_noise

        temporal_reg_w = float(self.opt['train'].get('temporal_backbone_reg_w', 0.0) or 0.0)
        if temporal_reg_w > 0 and self.temporal_reg_pred is not None:
            l_temporal = F.smooth_l1_loss(self.temporal_reg_pred, self.temporal_reg_target) * temporal_reg_w
            loss_dict['l_temporal'] = l_temporal
            l_g_total += l_temporal

        if self.lpips_loss_weight > 0 and self.lpips is not None:
            lpips_val = self.lpips(x_recon_cs, x_start)
            if isinstance(lpips_val, (list, tuple)):
                lpips_val = lpips_val[0]
            l_lpips = lpips_val.mean() * self.lpips_loss_weight
            loss_dict['l_g_lpips'] = l_lpips
            l_g_total += l_lpips

        if self.luma_regularizer_weight > 0 and self.lr_luma_context is not None:
            pred_luma_mean = self._rgb_to_luma_mean(x_recon_cs)
            target_luma_mean = self.lr_luma_context.to(pred_luma_mean.device)
            l_brightness = F.mse_loss(pred_luma_mean, target_luma_mean) * self.luma_regularizer_weight
            loss_dict['l_g_brightness'] = l_brightness
            l_g_total += l_brightness

        temporal_consistency = self._compute_temporal_consistency()
        if temporal_consistency is not None:
            l_temporal_consistency = temporal_consistency * self.temporal_consistency_weight
            loss_dict['l_g_temporal_consistency'] = l_temporal_consistency
            l_g_total += l_temporal_consistency

        l_g_total.backward()
        self.optimizer_g.step()
        if self.optimizer_temporal is not None:
            self.optimizer_temporal.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)
        self._record_training_history(current_iter, self.log_dict)

    # ====== 推理 ====== #
    def test(self):
        with torch.no_grad():
            self.bare_model.eval()
            if self.temporal_backbone is not None:
                self.temporal_backbone.eval()
            self.output = self.bare_model.ddim_LighT_sample(
                self.LR,
                structure=self.opt['val'].get('structure'),
                continous=self.opt['val'].get('ret_process', False),
                ddim_timesteps=self.opt['val'].get('ddim_timesteps', 50),
                return_pred_noise=self.opt['val'].get('return_pred_noise', False),
                return_x_recon=self.opt['val'].get('ret_x_recon', False),
                ddim_discr_method=self.opt['val'].get('ddim_discr_method', 'uniform'),
                ddim_eta=self.opt['val'].get('ddim_eta', 0.0),
                pred_type=self.opt['val'].get('pred_type', 'noise'),
                clip_noise=self.opt['val'].get('clip_noise', False),
                save_noise=self.opt['val'].get('save_noise', False),
                color_gamma=self.opt['val'].get('color_gamma', None),
                color_times=self.opt['val'].get('color_times', 1),
                return_all=self.opt['val'].get('ret_all', False),
                fine_diffV2=self.opt['val'].get('fine_diffV2', False),
                fine_diffV2_st=self.opt['val'].get('fine_diffV2_st', 200),
                fine_diffV2_num_timesteps=self.opt['val'].get('fine_diffV2_num_timesteps', 20),
                do_some_global_deg=self.opt['val'].get('do_some_global_deg', False),
                use_up_v2=self.opt['val'].get('use_up_v2', False)
            )
            self.bare_model.train()
            if self.temporal_backbone is not None:
                self.temporal_backbone.train()

            # 还原 padding
            if hasattr(self, 'pad_left') and not self.opt['val'].get('ret_process', False):
                self.output = pad_tensor_back(self.output, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
                self.LR = pad_tensor_back(self.LR, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
                self.HR = pad_tensor_back(self.HR, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)

    # ====== 分布式验证 ====== #
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    # ====== 适配历史论文拆分（可忽略） ====== #
    def find_lol_dataset(self, name):
        if name[0] == 'r':
            return 'SYNC'
        elif name[0] == 'n' or name[0] == 'l':
            return 'REAL'
        else:
            return 'LOL'

    # ====== 非分布式验证（含保存视频/三联图/metrics） ====== #
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']

        # 兼容两种 metrics 配置位置：val.metrics 或 根 metrics
        metrics_cfg = None
        if self.opt.get('val', {}).get('metrics') is not None:
            metrics_cfg = self.opt['val']['metrics']
        elif self.opt.get('metrics') is not None:
            metrics_cfg = self.opt['metrics']

        with_metrics = metrics_cfg is not None

        if self.opt['val'].get('fix_seed', False):
            next_seed = np.random.randint(10000000)
            get_root_logger().info(f'next_seed={next_seed}')

        if self.opt['val'].get('ret_process', False):
            with_metrics = False

        if with_metrics:
            self.metric_results = {metric: 0 for metric in metrics_cfg.keys()}

        # results 路径兜底
        if 'path' in self.opt and 'results' not in self.opt['path']:
            self.opt['path']['results'] = osp.join(
                self.opt['path'].get('experiments_root', '.'), 'results'
            )
        os.makedirs(self.opt['path']['results'], exist_ok=True)

        metric_data = dict()
        metric_data_pytorch = dict()
        pbar = tqdm(total=len(dataloader), unit='image')

        if self.opt['val'].get('split_log', False):
            self.split_results = {
                'SYNC': {metric: 0 for metric in (metrics_cfg or {}).keys()},
                'REAL': {metric: 0 for metric in (metrics_cfg or {}).keys()},
                'LOL':  {metric: 0 for metric in (metrics_cfg or {}).keys()},
            }

        # 统一视频写入管理器
        video_sink = self._VideoSink(self.opt['path']['results'])

        idx = -1  # 供 finally 的均值除数使用
        try:
            for idx, val_data in enumerate(dataloader):
                if self.opt['val'].get('fix_seed', False):
                    from basicsr.utils import set_random_seed
                    set_random_seed(0)

                if (not self.opt['val'].get('cal_all', False)) and \
                   (not self.opt['val'].get('cal_score', False)) and \
                   int(self.opt['ddpm_schedule']['n_timestep']) >= 4 and idx >= 3:
                    break

                # 文件名（用于三联图）
                img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

                # 前向
                self.feed_data(val_data)
                self.test()
                visuals = self.get_current_visuals()
                sr_img = tensor2img([visuals['sr']], min_max=(-1, 1))  # RGB uint8
                gt_img = tensor2img([visuals['gt']], min_max=(-1, 1))
                lq_img = tensor2img([visuals['lq']], min_max=(-1, 1))

                # 三联图
                if save_img:
                    if self.opt['is_train']:
                        save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                 f'{img_name}_{current_iter}.png')
                    else:
                        if self.opt['val'].get('suffix'):
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["val"]["suffix"]}.png')
                        else:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["name"]}.png')
                    if idx < self.opt['val'].get('show_num', 3) or self.opt['val'].get('show_all', False):
                        os.makedirs(osp.dirname(save_img_path), exist_ok=True)
                        imwrite(np.concatenate([lq_img, sr_img, gt_img], axis=1), save_img_path)

                # ====== 合成视频（仅 SR）=======
                if self.opt['val'].get('save_video', False):
                    # 1) 用“源 mp4 文件名（去扩展名）”作为视频分组键
                    src_path = val_data['lq_path'][0] if isinstance(val_data['lq_path'], (list, tuple)) \
                        else val_data['lq_path']
                    group_key = osp.splitext(osp.basename(src_path))[0]   # e.g. video02_003

                    # 2) 读取 FPS（失败回退 25）
                    fps = 25.0
                    try:
                        cap = cv2.VideoCapture(src_path)
                        if cap.isOpened():
                            _fps = cap.get(cv2.CAP_PROP_FPS)
                            if _fps and _fps > 0:
                                fps = float(_fps)
                        cap.release()
                    except Exception:
                        pass

                    # 3) 规范帧：确保 uint8 RGB、内存连续
                    sr_img = np.asarray(sr_img, dtype=np.uint8)         # RGB
                    sr_img = np.ascontiguousarray(sr_img)
                    h, w = sr_img.shape[:2]

                    # 4) 初始化 writer（只做一次），并保存第一帧 PNG 用于颜色对照
                    first_time = (group_key not in video_sink.cv2_writers) and (group_key not in video_sink.imageio_writers)
                    if first_time:
                        video_sink.get(group_key, w, h, fps)
                        # 探针：第一帧 PNG（RGB）便于和视频帧肉眼对比颜色
                        probe_png = osp.join(self.opt['path']['results'], f"{group_key}__probe.png")
                        imwrite(sr_img, probe_png)

                    # 5) 如果后续帧尺寸与首帧不一致，统一到首帧尺寸（防止容器损坏）
                    exp_w, exp_h = video_sink.sizes[group_key]
                    if (w, h) != (exp_w, exp_h):
                        sr_img = cv2.resize(sr_img, (exp_w, exp_h), interpolation=cv2.INTER_LINEAR)

                    # 6) 写入一帧（VideoSink 内部负责 RGB/BGR 的差异）
                    video_sink.write(group_key, sr_img)

                # ====== metrics ======
                if with_metrics:
                    metric_data['img'] = sr_img
                    metric_data['img2'] = gt_img
                    metric_data_pytorch['img'] = self.output
                    metric_data_pytorch['img2'] = self.HR

                    for name, opt_ in metrics_cfg.items():
                        if 'lpips' in name and self.lpips_bare_model is not None:
                            opt_ = dict(opt_)
                            opt_['device'] = self.device
                            opt_['model'] = self.lpips_bare_model

                        # 兼容 pytorch 张量型度量 & numpy 图像型度量
                        if 'pytorch' in opt_.get('type', ''):
                            val = calculate_metric(metric_data_pytorch, opt_).item()
                        else:
                            val = calculate_metric(metric_data, opt_)
                        if self.opt['val'].get('split_log', False):
                            self.split_results[self.find_lol_dataset(img_name)][name] += val
                        self.metric_results[name] += val

                torch.cuda.empty_cache()
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

                if self.opt['val'].get('cal_score_num', None) and \
                   idx >= self.opt['val']['cal_score_num']:
                    break

        finally:
            # 关闭所有视频句柄（必须，避免 MP4 损坏）
            video_sink.close_all()
            pbar.close()

        # 打印与记录 metrics
        if with_metrics and idx >= 0:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        if self.opt['val'].get('cal_score', False):
            import sys
            sys.exit()

        if self.opt['val'].get('fix_seed', False):
            from basicsr.utils import set_random_seed
            set_random_seed(next_seed)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        logger = get_root_logger()
        log_str = f'Validation {dataset_name} @ iter {current_iter}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger.info(log_str)
        self._record_validation_history(current_iter, dataset_name, self.metric_results)

        if self.opt['val'].get('split_log', False):
            for ds_name, num in zip(['LOL', 'REAL', 'SYNC'], [15, 100, 100]):
                if ds_name in getattr(self, 'split_results', {}) and num > 0:
                    log_str = f'Validation {ds_name}\n'
                    for metric, value in self.split_results[ds_name].items():
                        log_str += f'\t # {metric}: {value/num:.4f}\n'
                    logger.info(log_str)
                    self._record_validation_history(current_iter, ds_name, {k: v / num for k, v in self.split_results[ds_name].items()})

        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        # 对齐尺寸
        if self.LR.shape != self.output.shape:
            self.LR = F.interpolate(self.LR, self.output.shape[2:])
            self.HR = F.interpolate(self.HR, self.output.shape[2:])
        out_dict['gt'] = self.HR.detach().cpu()
        out_dict['sr'] = self.output.detach().cpu()

        lq_src = getattr(self, 'lq_vis', None)
        if lq_src is None:
            lq_src = self.LR
        if lq_src.shape[1] > 3:
            lq_src = lq_src[:, :3, :, :]
        out_dict['lq'] = lq_src.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network([self.ddpm], 'net_g', current_iter, param_key=['params'])
