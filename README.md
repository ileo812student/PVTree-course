<!-- SPDX-License-Identifier: Apache-2.0 -->

# PVTree 掌静脉合成与风格重分配课程项目

基于开源 PVTree 与 PCE-Palm 的课程实验代码。项目先生成三维血管树及二维投影，
再通过预训练 PCE 将血管图案转换成掌静脉图像，最后用合成图像训练识别器，
在真实 CASIA 940 nm 开发集上评价识别表现。

本仓库是**核心源码提交版**，仅包含代码、运行配置、测试、本 README 及原始许可证。
不包含真实生物特征数据、合成图片、模型权重、latent 文件、训练结果、过程报告、论文或 PPT。
下载源码不会自动下载数据，也不会自动启动生成或训练。

## 研究内容

- **PVTree 生成**：不同基础血管树定义不同合成身份，每个身份生成 7 张投影样本。
- **光学扰动**：独立离线照明、轻微高斯模糊和弱高斯噪声，支持固定种子和逐图记录。
  这些操作是成像条件近似，不是经过物理验证的皮肤散射模型。
- **风格分配对照**：预先生成同一份 7,000 个八维 latent。
  A0 每身份使用同一来源中心附近的 7 个向量；C1 每身份使用来自 7 个不同中心的向量。
  两组向量值、输入图案与身份标签不变，每个向量各使用一次。
- **可复现 PCE 推理**：单图统计兼容模式、关闭 Dropout、按相对路径派生稳定噪声种子，
  保留原噪声分布并隔离随机状态。纯 eval 暗输出问题没有通过重训或重估 BN 修复。
- **识别验证**：ResNet18 + 256 维 embedding + ArcFace，20 epochs，三个训练种子成对比较。
  生成器不重新训练；训练识别器是检验合成数据用途的手段。

课程实验的最终主要 EER 门槛未通过，低 FAR 下 TAR 有探索性积极信号，
不宣称独立真实测试提升。开发集已多轮使用，latent 也未被证明是纯风格变量。

## 项目结构

```text
.
├── main.py                    # 血管树与多视角图案生成入口
├── get_bezier.py              # Bezier 血管参数生成
├── args.py                    # 血管生成参数
├── kamiya.py, class_v_s.py     # 血管生长与结构
├── tools.py, data_tools.py     # 几何、投影和图像工具
├── crop_veinline.py            # 血管图案裁剪
├── extract_pceline.py          # 原工程图案处理工具
├── pcemodel/
│   ├── style_generate.py       # manifest 驱动的确定性配对推理
│   ├── test.py, run_gen.sh     # 原始推理入口（非最终配对协议）
│   ├── models/                # PCE 网络及 checkpoint 加载
│   ├── data/, options/, util/ # 输入预处理、选项与工具
│   └── train.py, train.sh      # 上游原始训练入口，本课程不重训 PCE
├── style_reassignment.py       # latent 采样、A0/C1 分配和完整性检查
├── optical_perturb.py          # 可独立运行的离线光学扰动
├── verify_optical_ab.py        # 光学 A/B 完整性核查
├── casia_prepare.py            # 本地授权 CASIA 的 940 nm ROI 准备
├── recognition/
│   ├── model.py               # ResNet18 embedding 与 ArcFace
│   ├── data.py                # 身份目录及配对数据读取
│   ├── train.py               # CUDA/AMP 训练、dev 选模和日志
│   ├── evaluate.py            # 固定 checkpoint 的配对评测
│   └── metrics.py             # ROC、EER 与 TAR@FAR
├── configs/                   # 光学配置、早期 A/B 及最终六次训练配置
├── tests/                     # 无外部数据的回归测试
├── requirements.txt           # 依赖版本
├── pytest.ini
├── .gitignore                 # 数据、权重和实验资料的忽略规则
├── LICENSE                    # 上游 Apache-2.0 许可证
└── README.md
```

`datasets/`、`experiments/`、`pcemodel/checkpoints/` 等仅是运行时本地目录，不随 Git 提交。
身份图片采用 `<identity>/<identity>_sample0.png` 至 `sample6.png`，
如 `c1_0/c1_0_sample0.png`；`c1` 到 `c4` 是血管主干类别，不是 CASIA 身份。

## 环境安装

已使用 WSL、Python 3.10、PyTorch 2.6.0+cu124 和 NVIDIA RTX 4060 系列 GPU 运行。
PCE 配对推理与正式识别配置要求 CUDA；不在 CUDA 不可用时退回 CPU 正式训练。
已有可用环境无需重装。

```bash
conda create -n pvtree python=3.10 pip -y
conda activate pvtree
python -m pip install setuptools==80.9.0
python -m pip install --no-build-isolation -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

依赖文件锁定 CUDA 12.4 版 PyTorch；其他硬件需要自行选择兼容构建，不保证跨 GPU、
跨 PyTorch 版本生成图片逐字节一致。

## 数据与权重准备

真实 CASIA 数据需根据数据库授权自行获得，不提供重新分发下载，也不随仓库上传。
合成身份由 PVTree 的基础血管树定义，并非由 CASIA dev 的 100 个掌身份逐一合成。
CASIA dev 的 100 个掌身份、600 张图片用于识别开发评价和 checkpoint 选择，不做梯度训练。
ROI 是否解剖对齐、编号是否代表独立采集 session，不能仅凭程序运行成功来确认。

PCE 使用原项目提供的官方权重，不在课程流程中重新训练。下载说明来源：
[PVTree 原仓库](https://github.com/Sunniva-Shang/PVTree-palmvein-generation)，
[原 README 提供的 checkpoint](https://drive.google.com/file/d/1aN7Bi-dDXWC_EpwmMiK_iqXCw2hyyK8h/view?usp=share_link)。
下载后按上游文件结构放置：

```text
pcemodel/checkpoints/palm_vein_128_black/latest_net_G.pth
pcemodel/checkpoints/palm_vein_128_black/latest_net_E.pth
```

## 主要使用方式

以下是复现入口说明；全量操作需要自行准备 GPU、权重和授权数据。
请先用少量样本检查，输出使用新目录，不覆盖已有实验。

### 1. 生成血管图案

```bash
python get_bezier.py --num_ids 1000 --nproc 4 --seed 2026   --output experiments/pvtree_1000/patterns/bezier
python main.py --num-identities 1000 --samples-per-identity 7 --workers 4   --seed 2026 --output-root experiments/pvtree_1000/patterns   --bezier-root experiments/pvtree_1000/patterns/bezier
```

### 2. 创建 latent 与 A0/C1 分配

```bash
python style_reassignment.py prepare   --input experiments/pvtree_1000/patterns/palmvein   --output experiments/style_reassignment/latent   --latent-seed 2026 --reassignment-seed 2027 --radius 0.5
```

该命令创建 latent pool、原始与重分配映射及审计记录，不下载权重、不训练模型。

### 3. 用同一推理条件生成 A0/C1

```bash
for group in A0 C1; do
  python pcemodel/style_generate.py     --dataroot experiments/pvtree_1000/patterns/palmvein     --name palm_vein_128_black --checkpoints_dir pcemodel/checkpoints     --assignment_manifest experiments/style_reassignment/latent/assignments_${group}.jsonl     --latent_pool experiments/style_reassignment/latent/latent_pool.npy     --results_dir experiments/pce_repro_final/data/${group}     --inference_seed 2028 --generator_mode batch_stats_no_dropout     --sample_noise_mode path_sha256_v1 --batch_size 1 --eval     --preprocess resize --load_size 128 --crop_size 128 --no_flip     --input_nc 1 --output_nc 1 --nz 8 --use_dropout --upsample bilinear     --netG unet_128 --netD basic_128_multi --netE resnet_128 --dataset_mode single
done
```

虽然结构加载使用 `--use_dropout`，兼容模式会显式关闭实际 Dropout。
生成器相关 BatchNorm 使用单图统计且不更新 running statistics；E 保持 eval，
此生成路径不调用编码器。按相对路径派生噪声种子，使同路径的 A0/C1 噪声配对。

### 4. 识别训练与开发集评测

`configs/recognition_A0_seed2026.json` 等六份配置保留最终相同训练设置，路径为仓库相对路径。
配置使用 20 epochs、batch 128、AdamW、AMP，最低 dev EER 选模。
已完成实验不需要再次运行；以下是独立复现时的命令形式：

```bash
python -m recognition.train --config configs/recognition_A0_seed2026.json
python -m recognition.train --config configs/recognition_C1_seed2026.json
# 其他两对使用 seed2027、seed2028 对应配置。
python -m recognition.evaluate   --checkpoint experiments/pce_repro_final/recognition/seed_2026/A0/checkpoint_best_dev.pt   --pairs datasets/casia_multi_940_roi/manifests/casia_dev_pairs.csv   --output experiments/course_dev_eval/A0_seed2026 --batch-size 128 --num-workers 4
```

配置中的 test 路径仅是既有配置字段，以上命令不评测 test。
最终阶段不使用 CASIA test 选参数、选 checkpoint 或运行评测。

### 5. 可选的独立光学扰动

光学扰动是早期独立对照，**不叠加到最终 A0/C1 风格分配实验**。
输入必须是最终 PCE 图像，不是白底血管线图。

```bash
python optical_perturb.py --input experiments/pce_repro_final/data/A0   --output experiments/optical_demo --config configs/optical_mild.json   --seed 2026 --max-images 32
```

`none` 按原文件复制；`mild`、`moderate` 和单算子配置提供不同保守起点，
默认参数未经过真实采集物理校准，不保证提升识别性能。

## 测试

无需下载真实数据或预训练权重，测试用例会创建临时人工样本：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
python -m pip check
```

覆盖血管生成、PCE 网络前向与噪声 RNG 隔离、光学 none/复现/格式、身份映射、
latent 一次性分配、CASIA ROI 工具及识别指标人工样例。
单元测试不替代完整生成与真实域性能验证。

## 来源与许可证

- 上游：[PVTree-palmvein-generation](https://github.com/Sunniva-Shang/PVTree-palmvein-generation)。
- PCE 部分借鉴：[PCE-Palm](https://github.com/Ukuer/PCE-Palm)。
- 保留原仓库的 Apache-2.0 [LICENSE](LICENSE) 与源代码中的原始声明。
- 课程扩展包括离线光学扰动、latent 分配、路径级随机噪声控制、识别实验及回归测试；
  不将上游模型或算法声称为原创。使用与引用应遵守各上游项目和数据集的要求。

### 许可范围与权利归属

本仓库的课程新增源码、测试、配置及说明文档同样采用 **Apache License 2.0**，
除非某文件已有另行声明。课程新增部分及修改部分的版权归相应课程贡献者所有
（`Copyright 2026 PVTree-course contributors`）；这一声明不覆盖或转移上游代码的版权。
原有版权、许可证和归属声明保留不变，不表示原作者认可或担保本课程扩展。

使用者可在遵守 Apache-2.0 条件的前提下使用、修改和再分发，包括商业使用。
再分发时须附带许可证、保留适用的原始声明，并在修改过的文件中显著说明修改；
若相关上游分发包含 NOTICE，须按许可证要求保留其中适用的归属信息。
代码按许可证约定以现状提供，不保证识别性能、适销性或特定用途适用性。

以上许可仅针对仓库中相应的代码与文档，不自动授予 CASIA 等数据集、预训练权重、
论文图片、商标或外部依赖的使用和再分发权利；它们应分别遵守自身条款。
课程实验或学术使用仍应引用实际使用的原始工作。

### 上游修改声明

本 README 已由 PVTree-course 贡献者于 2026 年改写，增加课程介绍、源码结构、
复现入口与许可说明，不是上游原版 README。
源码对照基线为 PVTree 上游提交
[`5eaf80fab1ee8a263dde990e6c6dd561bccfceee`](https://github.com/Sunniva-Shang/PVTree-palmvein-generation/commit/5eaf80fab1ee8a263dde990e6c6dd561bccfceee)。
2026-10-28 补充以下修改文件的显著声明；该次补充不改变算法、配置或实验结果：

| 修改文件 | 相对上游的课程改动 |
|---|---|
| `args.py` | 生成数量、路径、种子、并行和恢复参数；浮点类型及显式 argv |
| `data_tools.py` | 稳定遍历、身份子目录、路径级增强种子和覆盖保护 |
| `get_bezier.py` | 身份级种子、恢复和覆盖保护 |
| `main.py` | 可复现身份生成、投影恢复、可配置并行及配置/状态记录 |
| `pcemodel/options/test_options.py` | 推理种子和恢复选项 |
| `pcemodel/run_gen.sh` | 入口路径解析及严格错误处理 |
| `pcemodel/run_vein.sh` | 输入检查、路径适配、输出/种子/GPU 参数和恢复支持 |
| `pcemodel/test.py` | 种子、可选 eval、latent dtype、尾组处理及输出保护 |
| `requirements.txt` | 锁定验证环境版本并补充课程依赖 |
| `README.md` | 重写为课程源码提交版的项目、使用与许可说明 |

其他课程新增文件包括 `optical_perturb.py`、`verify_optical_ab.py`、
`style_reassignment.py`、`casia_prepare.py`、`pcemodel/style_generate.py`、
`recognition/`、`configs/`、`tests/`、`.gitignore` 和 `pytest.ini`。
这些是相对上述 PVTree 基线的课程新增实现或配置，不表示其中涉及的基础算法、
ResNet、ArcFace 或所调用的第三方库是课程原创。
