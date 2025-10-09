# tools/infer_video_temporalse.py
import os
import cv2
import torch
import numpy as np
from basicsr.utils.options import parse_options
from basicsr.utils import tensor2img
from lightdiff.models.lightdiff_model import LighTDiff

def bgr_to_normed_tensor(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).permute(2,0,1).float() / 255.0
    return t * 2.0 - 1.0  # [-1,1]

def clamp_indices(start, end, N):
    return [min(max(i, 0), N-1) for i in range(start, end)]

@torch.no_grad()
def infer_one_clip(opt, mp4_in, mp4_out, t_len=5, device='cuda'):
    # 构建模型
    model = LighTDiff(opt)
    model.ddpm.eval()
    if getattr(model, 'temporal_se', None) is not None:
        model.temporal_se.eval()

    # 读取视频
    cap = cv2.VideoCapture(mp4_in)
    assert cap.isOpened(), f'Cannot open {mp4_in}'
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    ok = True
    while ok:
        ok, frm = cap.read()
        if ok:
            frames.append(frm)
    cap.release()
    assert len(frames) > 0, 'No frames read'

    H, W = frames[0].shape[:2]
    os.makedirs(os.path.dirname(mp4_out), exist_ok=True)
    writer = cv2.VideoWriter(mp4_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    half = t_len // 2
    N = len(frames)
    for t in range(N):
        ids = clamp_indices(t - half, t + half + 1, N)  # 长度 t_len
        lr_seq = torch.stack([bgr_to_normed_tensor(frames[i]) for i in ids], dim=0)  # [T,3,H,W]
        # feed_data 接受 [T,3,H,W] & [3,H,W]，HR_mid 仅占位
        model.feed_data({'LR_seq': lr_seq, 'HR_mid': lr_seq[half]})
        model.test()
        sr = model.get_current_visuals()['sr'][0]  # [3,H,W], cpu
        sr_img = tensor2img([sr], min_max=(-1, 1))  # HxWx3 uint8, RGB
        writer.write(sr_img[..., ::-1])  # BGR

    writer.release()
    print(f'[OK] saved: {mp4_out}')

if __name__ == '__main__':
    # 用 test 配置跑推理
    opt, _ = parse_options(is_train=False)
    infer = opt.get('infer', {})
    mp4_in = infer.get('mp4_in')
    mp4_out = infer.get('mp4_out')
    t_len = opt.get('datasets', {}).get('val', {}).get('clip_len', 5)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    infer_one_clip(opt, mp4_in, mp4_out, t_len=t_len, device=device)
