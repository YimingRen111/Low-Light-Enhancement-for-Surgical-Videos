# lightdiff/data/lol_dataset.py
import os
import glob
import random
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from basicsr.utils.registry import DATASET_REGISTRY

def _bgr_to_normed_tensor(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    return t * 2.0 - 1.0  # [-1,1]

def _clamp_indices(start, end, N):
    return [min(max(i, 0), N - 1) for i in range(start, end)]

@DATASET_REGISTRY.register()
class VideoWindowMP4Dataset(Dataset):
    """
    目录结构（同名配对，支持前/后缀规范化）：
      root_lr/
        [lr_prefix]name[lr_suffix].mp4
      root_hr/
        [hr_prefix]name[hr_suffix].mp4

    要求：同名（在去除前/后缀后的基名）一一对应、帧数一致、时间同步。
    必要 opt:
      - root_lr, root_hr
    可选 opt（用于命名规范化）：
      - lr_prefix: "", lr_suffix: ""
      - hr_prefix: "", hr_suffix: ""
    其他：
      - clip_len（奇数，默认 5）、center_index（默认 clip_len//2）
      - （可选）crop_size: [h, w]；hflip/vflip
      - name / phase 供 BasicSR 使用
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.root_lr = opt['root_lr']
        self.root_hr = opt['root_hr']
        self.clip_len = int(opt.get('clip_len', 5))
        assert self.clip_len % 2 == 1, 'clip_len 必须为奇数，例如 5'
        self.center = int(opt.get('center_index', self.clip_len // 2))
        assert 0 <= self.center < self.clip_len, 'center_index 越界'
        self.crop_size = opt.get('crop_size', None)   # [ch, cw] 或 None
        self.hflip = bool(opt.get('hflip', True))
        self.vflip = bool(opt.get('vflip', False))

        # 命名规范化参数（可选）
        self.lr_prefix = opt.get('lr_prefix', "")
        self.lr_suffix = opt.get('lr_suffix', "")
        self.hr_prefix = opt.get('hr_prefix', "")
        self.hr_suffix = opt.get('hr_suffix', "")

        # 收集 mp4
        lr_list = sorted(glob.glob(os.path.join(self.root_lr, '*.mp4')))
        hr_list = sorted(glob.glob(os.path.join(self.root_hr, '*.mp4')))
        assert len(lr_list) > 0 and len(hr_list) > 0, \
            f'LR/HR 为空: {self.root_lr} / {self.root_hr}'

        def stem(p):  # 文件名去扩展名
            return os.path.splitext(os.path.basename(p))[0]

        def normalize(name: str, pref: str, suff: str) -> str:
            """去掉前/后缀，得到用于配对的规范化基名"""
            if pref and name.startswith(pref):
                name = name[len(pref):]
            if suff and name.endswith(suff):
                name = name[:-len(suff)]
            return name

        # 建立规范化名称到路径的映射
        lr_map = {normalize(stem(p), self.lr_prefix, self.lr_suffix): p for p in lr_list}
        hr_map = {normalize(stem(p), self.hr_prefix, self.hr_suffix): p for p in hr_list}

        # 名称集合对齐检查
        names_lr = set(lr_map.keys())
        names_hr = set(hr_map.keys())
        missing_lr = sorted(list(names_hr - names_lr))
        missing_hr = sorted(list(names_lr - names_hr))
        assert not missing_lr and not missing_hr, \
            (f'同名（规范化后）配对失败：\n'
             f'  HR 中存在但 LR 缺失: {missing_lr}\n'
             f'  LR 中存在但 HR 缺失: {missing_hr}\n'
             f'  请检查 lr_prefix/lr_suffix 与 hr_prefix/hr_suffix 配置或修正数据命名。')

        # 按规范化名字排序建立配对
        names = sorted(list(names_hr))
        self.lr_paths = [lr_map[n] for n in names]
        self.hr_paths = [hr_map[n] for n in names]

        # 预读帧数 & 生成所有中心帧索引
        self._meta = []     # [(lr_path, hr_path, num_frames)]
        self._samples = []  # [(vid_idx, t)]
        half = self.clip_len // 2
        for i, (lp, hp) in enumerate(zip(self.lr_paths, self.hr_paths)):
            n_lr = self._probe_nframes(lp)
            n_hr = self._probe_nframes(hp)
            assert n_lr == n_hr and n_lr >= self.clip_len, f'帧数不匹配或不足: {lp} / {hp}'
            self._meta.append((lp, hp, n_lr))
            for t in range(half, n_lr - half):
                self._samples.append((i, t))

    @staticmethod
    def _probe_nframes(path):
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n

    @staticmethod
    def _read_all(path):
        cap = cv2.VideoCapture(path)
        frames = []
        ok = True
        while ok:
            ok, frm = cap.read()
            if ok:
                frames.append(frm)
        cap.release()
        return frames  # list of BGR uint8

    def __len__(self):
        return len(self._samples)

    def _sync_random_crop(self, lr_seq, hr_mid):
        if self.crop_size is None:
            return lr_seq, hr_mid
        _, _, H, W = lr_seq.shape
        ch, cw = int(self.crop_size[0]), int(self.crop_size[1])
        ch = min(ch, H); cw = min(cw, W)
        top = 0 if H == ch else random.randint(0, H - ch)
        left = 0 if W == cw else random.randint(0, W - cw)
        lr_seq = lr_seq[:, :, top:top+ch, left:left+cw]
        hr_mid = hr_mid[:, top:top+ch, left:left+cw]
        return lr_seq, hr_mid

    def _sync_flip(self, lr_seq, hr_mid):
        if self.hflip and random.random() < 0.5:
            lr_seq = torch.flip(lr_seq, dims=[3])  # 水平翻转（W维）
            hr_mid = torch.flip(hr_mid, dims=[2])
        if self.vflip and random.random() < 0.5:
            lr_seq = torch.flip(lr_seq, dims=[2])  # 垂直翻转（H维）
            hr_mid = torch.flip(hr_mid, dims=[1])
        return lr_seq, hr_mid

    def __getitem__(self, index):
        vid_idx, t = self._samples[index]
        lr_path, hr_path, N = self._meta[vid_idx]

        # 5s 很短，直接整段读入（避免频繁 seek）
        lr_frames = self._read_all(lr_path)
        hr_frames = self._read_all(hr_path)

        half = self.clip_len // 2
        ids = _clamp_indices(t - half, t + half + 1, N)   # 长度 clip_len
        lr_seq = torch.stack([_bgr_to_normed_tensor(lr_frames[i]) for i in ids], dim=0)  # [T,3,H,W]
        hr_mid = _bgr_to_normed_tensor(hr_frames[t])                                      # [3,H,W]

        # 同步增强（训练启用，验证关闭）
        phase = self.opt.get('phase', 'train')
        if phase == 'train':
            lr_seq, hr_mid = self._sync_random_crop(lr_seq, hr_mid)
            lr_seq, hr_mid = self._sync_flip(lr_seq, hr_mid)

        return {
            'LR_seq': lr_seq,        # [T,3,H,W]
            'HR_mid': hr_mid,        # [3,H,W]
            'lq_path': lr_path,    # 供日志/保存名
        }
