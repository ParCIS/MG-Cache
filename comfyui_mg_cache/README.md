# HunyuanVideo 的 ComfyUI 专用节点

节点使用仓库中的 HunyuanVideo 采样器，固定去噪 50 步。默认尺寸为 640 × 480，65 帧，种子为 42，$\alpha=0.03$，$\beta=0.3$。示例工作流默认关闭对照。

## 远端安装和启动

下面的命令使用已有的 `HunyuanVideo` Conda 环境。保留仓库目录结构并将源码同步到远端。此例假设独立部署目录下的 `source` 是仓库根目录，里面包含 `MG-Cache/HunyuanVideo` 和 `comfyui_mg_cache`。

```bash
cd /path/to/deployment
git clone --depth 1 --branch v0.3.27 https://github.com/comfyanonymous/ComfyUI.git ComfyUI
"$HOME/anaconda3/envs/HunyuanVideo/bin/python" -m venv --system-site-packages .venv

# 此组合沿用已有环境的 PyTorch 与 FlashAttention，前端使用 1.14.6。
sed 's/comfyui-frontend-package==1.14.5/comfyui-frontend-package==1.14.6/' \
    ComfyUI/requirements.txt > requirements-comfyui.txt
printf 'torch==2.4.0\nnumpy==1.26.4\ntorchvision==0.19.0\ntorchaudio==2.4.0\n' > constraints.txt
.venv/bin/python -m pip install -c constraints.txt -r requirements-comfyui.txt 'scikit-image==0.25.2'
ln -s "$PWD/source/comfyui_mg_cache" ComfyUI/custom_nodes/comfyui_mg_cache

cd ComfyUI
MODEL_BASE="$HOME/HunyuanVideo/ckpts" CUDA_VISIBLE_DEVICES=4 \
    ../.venv/bin/python main.py --listen 127.0.0.1 --port 8198 --disable-auto-launch
```

将 `CUDA_VISIBLE_DEVICES` 改为空闲 H100 的编号。现有环境需包含与 PyTorch 匹配的 FlashAttention、Diffusers 0.31.0、Transformers 4.46.3、ImageIO 和 imageio-ffmpeg。节点在进程内保留采样器，建议使用独立 ComfyUI 实例。权重目录应包含 `hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt`、`hunyuan-video-t2v-720p/vae/`、`text_encoder/` 和 `text_encoder_2/`。

可从客户端通过 SSH 将服务映射到本机。

```bash
ssh -N -L 8198:127.0.0.1:8198 chenzesen@10.160.4.102
```

用支持 H.264 的浏览器打开 `http://127.0.0.1:8198`，导入 [示例工作流](workflows/hunyuanvideo_mg_cache.json)。输入提示词并排队运行。打开节点的「启用对照」开关后再次排队，节点会在同一张 GPU 上先执行完整计算，再执行 MG-Cache。关闭对照会清除面板中的上一组对照展示。

节点左侧为提示词、视频规格与采样、缓存与对照，右侧为运行结果。开启对照后，「视频预览」从上到下显示完整计算和 MG-Cache，「质量曲线」从上到下显示 PSNR 和 SSIM。结果较长时可在右侧滚动，使用页签切换到「缓存热图」。

## 参数和输出

- 宽高须为 16 的正倍数，帧数须为 $4n+1$。节点在排队前显示预计的锁页层缓存占用，默认约 14.18 GiB。此值是两组 BF16 缓存的大小。模型、去噪中间张量和 VAE 另需内存。
- 默认 `cfg_scale=1`，使用 HunyuanVideo 的蒸馏引导，固定 `embedded_guidance_scale=6`。提高 `cfg_scale` 会启用负向提示词和双批次 CFG，层缓存估算随之翻倍。负向提示词留空时，CFG 使用采样器的默认负向词。两条分支使用相同提示词、调度器设置和随机种子。采样器各自重建 CUDA 随机数生成器，并对去噪初始潜变量做 SHA-256 校验。
- 每次运行写入 `ComfyUI/output/mg_cache/<运行 ID>/`，其中包含 24 FPS 的 MP4 和 `report.json`。报告包含输入、有效负向提示词、噪声指纹、去噪耗时和 MG-Cache 缓存决策。对照报告还包含基线耗时、加速比和逐帧指标。
- 去噪耗时使用 CUDA 同步后的墙钟时间，从进入去噪循环前计到循环结束。包含调度器和 MG-Cache 首次填充层缓存的开销。权重加载、提示词编码、初始潜变量准备、VAE 解码、指标和 MP4 写入均在计时区间之外。两条分支都执行同一套计时流程，各使用报告所列的一次 50 步去噪，没有额外预跑。对照固定按基线、MG-Cache 的顺序执行。加速比等于基线去噪耗时除以 MG-Cache 去噪耗时。
- PSNR 和 SSIM 在编码前的 `[0, 1]` 浮点 RGB 解码帧上逐帧计算。PSNR 以三个通道的像素均方误差计算，$-10\log_{10}(\mathrm{MSE})$。SSIM 使用 11 × 11 高斯窗，$\sigma=1.5$，使用总体协方差，三个通道等权平均。全片指标是逐帧指标的算术平均。完全相同帧的 PSNR 显示为 ∞，报告中写为字符串 `"inf"`。任一帧为 ∞ 时，全片 PSNR 均值也为 ∞。

## 缓存记录的含义

整步复用用校准的线性补偿参数近似上一次 Transformer 输出。调度器在该步照常更新潜变量。其余步骤执行 Transformer，初始化步骤写入调制后的输入和注意力结果。后续步骤根据参考输入和层权重累积相对 L1 误差，命中时复用注意力结果。双流层命中会跳过 QKV 投影、Q/K 归一化、RoPE 和注意力计算。单流层先执行联合线性投影，命中会跳过其后的 Q/K 归一化、RoPE 和注意力计算。两类层仍执行 MLP 和输出投影。

面板显示 1–50 步和 1–60 层，报告中的 `step` 从 1 起算，`reused_layers` 从 0 起算，前 20 层为双流层。整步复用的步骤不计入层注意力命中次数。

## 远端验收记录

2026 年 11 月 8 日在 `10.160.4.102` 的单张 NVIDIA H100 PCIe 上验收。使用物理 GPU 7、Ubuntu 20.04.6、Python 3.10.18、PyTorch 2.4.0（CUDA 11.8）、FlashAttention 2.6.3、ComfyUI 0.3.27 和前端 1.14.6。

默认工作流使用 640 × 480、65 帧、50 步和种子 42。

| 模式 | 完整计算去噪 | MG-Cache 去噪 | 加速比 | PSNR 均值 | SSIM 均值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MG-Cache 单跑 | — | 98.42 s | — | — | — |
| 对照 | 148.77 s | 92.83 s | 1.60× | 28.60 dB | 0.9300 |

两次 MG-Cache 运行均记录 29 次整步复用和 184 次层注意力命中，输出 MP4 字节一致。对照两分支的初始潜变量 SHA-256 一致。FFmpeg 解码检查确认三段输出均为 640 × 480、65 帧、24 FPS。对照完成后，在同一进程切回单跑并修改到 160 × 96、9 帧、CFG 2，负向提示词生效且生成新规格视频。非法宽度 641 在采样前被拒绝。远端还通过了 Python 和前端语法检查、现有 CLI 的参数入口检查、合成帧指标检查和 CUDA 缓存预取检查。

Playwright 在远端导入工作流并验证了预估、对照开关、排队生成、逐帧曲线、热图和报告下载。关闭对照后，旧对照视图即时清除，随后单跑只展示一段视频和本次缓存热图。Chrome for Testing 134 成功播放默认对照的两段 H.264 视频。

远端部署目录为 `/home/chenzesen/mg-cache-comfyui.XxEuK8`。其中 `acceptance-environment.txt` 和 `acceptance-results.txt` 记录环境与测量数据，`output/playwright/default-comparison.png` 为默认对照界面截图。视频及逐帧报告位于部署目录下的 `ComfyUI/output/mg_cache/`，默认单跑目录为 `56c2c4499f9045dca518354a31351c90`，默认对照目录为 `d39e5579d12746f9afc291dd64b56b97`。
