# Name That LLM · Qwen3 LoRA 后训练

对公开的 [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) 做有监督微调。冻结预训练权重，在每层注意力的 query 和 value 投影上训练 rank-16 LoRA，并训练 18 类分类头，读取段落结束 token 的隐藏状态。第一轮仅使用官网训练集，先量化跨题目表现，再决定是否增加数据。输入只使用原文，保留空格、标点、大小写等可能反映写作风格的信号。长文本保留首尾。

按 `prompt_id` 分三折，同一题目的回答进入同一折。用每折验证集的 balanced accuracy 选 epoch，然后平均三折模型的测试概率。每折结果、汇总 OOF balanced accuracy、各类召回率和混淆矩阵写入 `metrics.json`。由于各折验证标签也用于选 epoch，OOF 分数可能略偏乐观，最终效果以竞赛隐藏测试为准。

## 数据

登录 [Kaggle 竞赛数据页](https://www.kaggle.com/competitions/name-that-llm/data)，接受规则并下载官方数据。解压后应有 `train.csv` 和 `test.csv`。可见的测试文件是流程调试用的示例，提交 Notebook 后 Kaggle 会换成完整隐藏测试集。

## 本地先测通，再跑完整验证

环境建议使用 Python 3.11 或 3.12，已装与显卡驱动匹配的 CUDA 版 PyTorch。本地 6 GB GPU 显存紧张时使用 `--batch-size 2 --accumulation 8 --max-length 256`。

```powershell
python -m pip install -r requirements.txt
python train.py --data ./data --output ./runs/quick --folds 2 --epochs 1 --batch-size 2 --max-length 128
python train.py --data ./data --output ./runs/full --batch-size 2 --accumulation 8 --max-length 256
```

命令应在仓库根目录运行。单跑验证可加 `--validate-only`。首次在线运行从 Hugging Face 下载模型。有本地权重时使用 `--model ./qwen3-base --offline`。每次完整实验用独立输出目录。

## CCDS TC2

在已有的集群 Python 环境中安装依赖，准备 HF 格式的 Qwen3 权重目录，把 `train.py`、`train.slurm` 和数据传到集群。激活环境，从脚本目录提交。第三个参数起可覆盖训练配置。

```bash
sbatch train.slurm ./data ./qwen3-base --validate-only
squeue -u "$USER"
```

作业日志为 `slurm-作业号.out`，结果在 `runs/作业号/`。Slurm 参数对应附件手册中的 TC2 A40 队列。

## NSCC ASPIRE 2A

将数据放在作业目录的 `data/`，将模型放在 `qwen3-base/`。按 NSCC 接入流程登录，检查项目余额，激活 CUDA 版 PyTorch 环境。提交时通过 `-P` 指定有余额的项目。

```bash
myprojects
qsub -P PROJECT_ID train.pbs
qstat -u "$USER"
```

若使用已有虚拟环境，可用 `qsub -P PROJECT_ID -v QWEN_PYTHON=/path/to/venv/bin/python train.pbs` 指定解释器。如果作业目录有 `deps/`，会优先从该目录加载 Python 依赖。训练日志为 `train-作业号.log`。

## Kaggle 提交

1. 导入 `notebooks/name_that_llm.ipynb`，添加 **Name That LLM** 竞赛数据，启用一张 GPU。
2. 添加公开的 Hugging Face 格式 Qwen3-0.6B-Base 权重。挂载目录应包含 `config.json`、`model.safetensors` 或 `pytorch_model.bin`，以及 `tokenizer.json` 和 `tokenizer_config.json`。可以将公开的 `Qwen/Qwen3-0.6B-Base` 模型文件放进自建 Kaggle 输入数据集，再挂载到 Notebook。第二个代码单元会尝试定位该目录，也支持手动填入 `MODEL_DIR`。
3. 关闭互联网，保存并运行 Notebook。成功后确认输出中存在 `/kaggle/working/submission.parquet`，再从运行完成的版本提交竞赛。运行过程中会打印每折每轮的验证分数。若超出比赛的 9 小时限制，可先使用 2 折、3 轮、256 token 重新验证耗时。

本方案用 Transformers 4.57.6 验证。如果 Kaggle 镜像中的版本不兼容，可以在 Kaggle 的 Dependency Manager 配置 `pip install transformers==4.57.6`，由 Kaggle 在离线 Notebook 运行前安装。

Notebook 会从竞赛输入目录读取 `test.csv`，依原顺序输出 `id` 和 `model`，并回读 Parquet 检查。Notebook 自包含训练代码。若修改独立脚本，需要同步 Notebook 的第一个代码单元。


## 公开语料与融合

从 [Name That LLM Open Pool](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool) 获取 `ntl_open_pool.parquet`。在仓库根目录运行：

```bash
python scripts/audit_pool.py --official-train data/train.csv --pool /path/to/ntl_open_pool.parquet --output data
python train.py --data data --output runs/qwen --model Qwen/Qwen3-0.6B-Base --external data/external_bulk.csv
python scripts/sparse_oof.py --data data --output runs/sparse --external external_bulk.csv --external-only
```

`notebooks/` 保存基线和各版融合。融合 Notebook 需要挂载对应基础权重和自行训练的 LoRA 权重。每组 LoRA 目录包含 `metrics.json` 和三个 `fold-*-lora.pt`，挂载目录名以 Notebook 内的查找逻辑为准。变更配置时，先验证 Kaggle 断网运行能生成 `submission.parquet`。

- [实验记录](docs/EXPERIMENTS.md)
- [Kaggle 源码 Notebook](https://www.kaggle.com/code/cilaaga001/name-that-llm-project-source)

## 协作

获得写入邀请后，协作者可从 `main` 建分支、推送改动并提交 Pull Request。其他贡献者可以先 fork。竞赛数据请从 Kaggle 获取，凭据和集群配置保存在各自的私有环境。
