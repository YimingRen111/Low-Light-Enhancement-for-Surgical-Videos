# Metrics

[English](Metrics.md) **|** [简体中文](Metrics_CN.md)

## PSNR and SSIM

## NIQE

## LPIPS

LPIPS (Learned Perceptual Image Patch Similarity) measures perceptual
distance between two images using deep features.  BasicSR exposes it via the
``calculate_lpips`` entry in the metric registry once the
[`lpips`](https://github.com/richzhang/PerceptualSimilarity) package has been
installed (``pip install lpips``).

To evaluate LPIPS during ``basicsr/test.py`` runs, add it to the ``metrics``
section of your testing/validation configuration:

```yml
val:
  metrics:
    lpips:
      type: calculate_lpips
      net: alex      # optional, defaults to "alex"
      use_gpu: true  # optional, defaults to True when CUDA is available
```

With the configuration in place, launch the test script as usual (for example
``python basicsr/test.py -opt options/test/HiFaceGAN/test_hifacegan.yml``) and
the LPIPS score will be reported alongside the other metrics.

### Using LPIPS inside LighTDiff

LighTDiff reuses BasicSR's metric registry, so the exact same ``metrics`` block
from above can be added to your LightDiff YAML configuration.  If you want to
reuse the LPIPS instance that LightDiff creates during model initialization,
set ``type: calculate_lpips_lol`` (the wrapper simply forwards to
``calculate_lpips`` with the cached model).  When running the LightDiff entry
points on Windows PowerShell, make sure both repositories are on
``PYTHONPATH`` before launching either ``train.py`` or ``test.py``:

```powershell
& conda 'shell.powershell' 'hook' | Out-String | Invoke-Expression
conda activate lightdiff
$env:PYTHONPATH='E:\ELEC5020\LighTDiff-main\LighTDiff-main\BasicSR;E:\ELEC5020\LighTDiff-main\LighTDiff-main\LighTDiff'
python LighTDiff\lightdiff\train.py -opt LighTDiff\configs\train_video_temporalse.yaml --force_yml name=longrun1_run
# For evaluation
python LighTDiff\lightdiff\test.py -opt LighTDiff\configs\test_video_temporalse.yaml
```

Any LPIPS entries under ``val.metrics`` (or the top-level ``metrics`` section)
will then be picked up automatically during training validation and standalone
testing.  After each run finishes, LightDiff prints the aggregated metric
values (PSNR, SSIM, LPIPS, etc.) to both the console and the generated log file
inside ``experiments/<run_name>/log`` so you can confirm LPIPS is being
calculated.

#### Where validation outputs are written

During both ``basicsr/test.py`` and ``lightdiff/test.py`` runs, the model
enters the shared ``validation`` routine defined on ``LighTDiff``.  When the
``save_img`` flag is enabled (``val.save_img: true`` in your YAML), every
mini-batch produces an ``[LQ | SR | GT]`` triptych that is saved under the
``visualization`` directory determined by ``make_exp_dirs``: for testing it is
``results/<run_name>/visualization/<dataset_name>/``; during training it becomes
``experiments/<run_name>/visualization/<dataset_name>/``.  You can see this in
``lightdiff_model.py`` where the filenames are assembled via
``os.path.join(self.opt['path']['visualization'], dataset_name, ...)`` and the
images are written with ``imwrite`` after concatenating the low-light input,
restored frame, and reference frame.  If ``val.save_video`` is true, the same
method also streams the restored frames into ``.mp4`` files in
``results/<run_name>/`` using the ``_VideoSink`` helper.

## FID

> FID measures the similarity between two datasets of images. It was shown to correlate well with human judgement of visual quality and is most often used to evaluate the quality of samples of Generative Adversarial Networks.
> FID is calculated by computing the [Fréchet distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance) between two Gaussians fitted to feature representations of the Inception network.

References

- https://github.com/mseitzer/pytorch-fid
- [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500)
- [Are GANs Created Equal? A Large-Scale Study](https://arxiv.org/abs/1711.10337)

### Pre-calculated FFHQ inception feature statistics

Usually, we put the downloaded inception feature statistics in `basicsr/metrics`.

:arrow_double_down: Google Drive: [metrics data](https://drive.google.com/drive/folders/13cWIQyHX3iNmZRJ5v8v3kdyrT9RBTAi9?usp=sharing)
:arrow_double_down: 百度网盘: [评价指标数据](https://pan.baidu.com/s/10mMKXSEgrC5y7m63W5vbMQ) <br>

| File Name         | Dataset | Image Shape    | Sample Numbers|
| :------------- | :----------:|:----------:|:----------:|
| inception_FFHQ_256-0948f50d.pth | FFHQ | 256 x 256 | 50,000 |
| inception_FFHQ_512-f7b384ab.pth | FFHQ | 512 x 512 | 50,000 |
| inception_FFHQ_1024-75f195dc.pth | FFHQ | 1024 x 1024 | 50,000 |
| inception_FFHQ_256_stylegan2_pytorch-abba9d31.pth | FFHQ | 256 x 256 | 50,000 |

- All the FFHQ inception feature statistics calculated on the resized 299 x 299 size.
- `inception_FFHQ_256_stylegan2_pytorch-abba9d31.pth` is converted from the statistics in [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch).
