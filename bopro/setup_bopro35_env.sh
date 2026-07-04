#!/bin/bash
# setup_bopro35_env.sh
# 创建 Python 3.11 新环境 bopro35，用于运行 Qwen3.5-4B
# 包含 SGLang + botorch + rdkit 等所有依赖

set -e

source /data/xk/zhaoyang/miniconda3/etc/profile.d/conda.sh

echo "=========================================="
echo "Step 1: 创建 conda 环境 bopro35 (Python 3.11)"
echo "=========================================="
conda create -n bopro35 python=3.11 -y
conda activate bopro35

echo "=========================================="
echo "Step 2: 安装 rdkit (conda-forge)"
echo "=========================================="
conda install -c conda-forge rdkit -y

echo "=========================================="
echo "Step 3: 安装 SGLang [srt] (含 torch 2.7.1 + transformers)"
echo "=========================================="
pip install 'sglang[srt]'

echo "=========================================="
echo "Step 4: 安装 botorch + gpytorch"
echo "=========================================="
pip install botorch gpytorch

echo "=========================================="
echo "Step 5: 安装 bopro 其他依赖"
echo "=========================================="
pip install scikit-learn pandas numpy scipy matplotlib seaborn
pip install wandb tqdm regex requests openai emoji nltk
pip install pybase64 orjson

echo "=========================================="
echo "Step 6: 验证安装"
echo "=========================================="
python -c "
import sys
print('Python:', sys.version)
import torch
print('torch:', torch.__version__)
import transformers
print('transformers:', transformers.__version__)
import botorch
print('botorch:', botorch.__version__)
import gpytorch
print('gpytorch:', gpytorch.__version__)
from rdkit import Chem
print('rdkit: OK')
from sglang.srt.entrypoints.http_server import launch_server
print('sglang: OK')
print('All imports successful!')
"

echo "=========================================="
echo "Step 7: 验证 Qwen3.5-4B 配置"
echo "=========================================="
export HF_ENDPOINT=https://hf-mirror.com
python -c "
from transformers import AutoConfig
try:
    c = AutoConfig.from_pretrained('Qwen/Qwen3.5-4B')
    print('Qwen3.5-4B model_type:', c.model_type)
    print('Qwen3.5-4B config loaded successfully!')
except Exception as e:
    print('ERROR:', e)
"

echo "=========================================="
echo "环境创建完成!"
echo "激活: conda activate bopro35"
echo "=========================================="
