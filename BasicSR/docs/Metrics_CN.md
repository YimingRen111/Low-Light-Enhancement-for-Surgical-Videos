# 评价指标

[English](Metrics.md) **|** [简体中文](Metrics_CN.md)

## PSNR and SSIM

## NIQE

## LPIPS

LPIPS（Learned Perceptual Image Patch Similarity）使用深度特征衡量两张图像
之间的感知距离。 在安装
[`lpips`](https://github.com/richzhang/PerceptualSimilarity) 库（``pip install
lpips``）后，可以通过 ``calculate_lpips`` 这个注册名在 BasicSR 中直接调用。

如果希望在运行 ``basicsr/test.py`` 时同步计算 LPIPS，只需在测试/验证配置
文件的 ``metrics`` 字段中加入相应条目：

```yml
val:
  metrics:
    lpips:
      type: calculate_lpips
      net: alex      # 可选，默认 "alex"
      use_gpu: true  # 可选，默认在可用时使用 CUDA
```

随后按常规方式执行测试脚本（例如
``python basicsr/test.py -opt options/test/HiFaceGAN/test_hifacegan.yml``），LPIPS
结果就会与其他评价指标一起输出。

### 在 LighTDiff 中使用 LPIPS

LighTDiff 同样复用了 BasicSR 的度量注册表，因此可以直接把上面的
``metrics`` 配置复制到 LightDiff 的 YAML 文件里。如果想复用模型初始化时
创建的 LPIPS 实例，可以把 ``type`` 设置为 ``calculate_lpips_lol``（该包裹
函数会把缓存的模型转发给 ``calculate_lpips``）。在 Windows PowerShell 中
运行训练或测试脚本时，务必先把 BasicSR 与 LighTDiff 两个目录加入
``PYTHONPATH``，再启动 ``train.py`` 或 ``test.py``：

```powershell
& conda 'shell.powershell' 'hook' | Out-String | Invoke-Expression
conda activate lightdiff
$env:PYTHONPATH='E:\ELEC5020\LighTDiff-main\LighTDiff-main\BasicSR;E:\ELEC5020\LighTDiff-main\LighTDiff-main\LighTDiff'
python LighTDiff\lightdiff\train.py -opt LighTDiff\configs\train_video_temporalse.yaml --force_yml name=longrun1_run
# 评测示例
python LighTDiff\lightdiff\test.py -opt LighTDiff\configs\test_video_temporalse.yaml
```

在 ``val.metrics``（或顶层 ``metrics``）里声明的 LPIPS 指标就会在训练期验证
以及独立测试脚本中自动计算。运行结束后，LightDiff 会在控制台和
``experiments/<运行名>/log`` 目录下生成的日志文件中同时打印 PSNR、SSIM、
LPIPS 等聚合指标，方便你确认 LPIPS 已成功参与评测。

#### 验证阶段的图片与视频输出目录

无论执行 ``basicsr/test.py`` 还是 ``lightdiff/test.py``，最终都会调用
``LighTDiff`` 的 ``validation`` 方法。当 YAML 中设置 ``val.save_img: true``
时，每个 mini-batch 都会拼接 ``[LQ | SR | GT]`` 三联图，并写入
``make_exp_dirs`` 创建的 ``visualization`` 路径：在测试阶段为
``results/<运行名>/visualization/<数据集名>/``，训练阶段则是
``experiments/<运行名>/visualization/<数据集名>/``。源码中通过
``os.path.join(self.opt['path']['visualization'], dataset_name, ...)`` 拼接文件
名，并在写入前将 LQ、SR、GT 图像沿宽度方向拼接。若 ``val.save_video``
为 true，同一流程还会借助 ``_VideoSink`` 辅助类把重建帧流式写入
``results/<运行名>/`` 下的 ``.mp4`` 文件，并额外保存第一帧 PNG 方便对照。

## FID

> FID measures the similarity between two datasets of images. It was shown to correlate well with human judgement of visual quality and is most often used to evaluate the quality of samples of Generative Adversarial Networks.
> FID is calculated by computing the [Fréchet distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance) between two Gaussians fitted to feature representations of the Inception network.

参考

- https://github.com/mseitzer/pytorch-fid
- [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500)
- [Are GANs Created Equal? A Large-Scale Study](https://arxiv.org/abs/1711.10337)

### Pre-calculated FFHQ inception feature statistics

通常, 我们把下载的 inception 网络的特征统计数据 (用于计算FID) 放在 `basicsr/metrics`.


:arrow_double_down: 百度网盘: [评价指标数据](https://pan.baidu.com/s/10mMKXSEgrC5y7m63W5vbMQ)
:arrow_double_down: Google Drive: [metrics data](https://drive.google.com/drive/folders/13cWIQyHX3iNmZRJ5v8v3kdyrT9RBTAi9?usp=sharing) <br>

| File Name         | Dataset | Image Shape    | Sample Numbers|
| :------------- | :----------:|:----------:|:----------:|
| inception_FFHQ_256-0948f50d.pth | FFHQ | 256 x 256 | 50,000 |
| inception_FFHQ_512-f7b384ab.pth | FFHQ | 512 x 512 | 50,000 |
| inception_FFHQ_1024-75f195dc.pth | FFHQ | 1024 x 1024 | 50,000 |
| inception_FFHQ_256_stylegan2_pytorch-abba9d31.pth | FFHQ | 256 x 256 | 50,000 |

- All the FFHQ inception feature statistics calculated on the resized 299 x 299 size.
- `inception_FFHQ_256_stylegan2_pytorch-abba9d31.pth` is converted from the statistics in [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch).
