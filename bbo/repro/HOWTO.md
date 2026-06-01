# 实验操作指南

## 前置信息

- 服务器: `ssh xk@180.209.6.83` (4× RTX 4090)
- 根目录: `/data/xk/zhaoyang`
- 日志目录: `/data/xk/zhaoyang/bbo_repro/logs`
- Conda环境: `ddom_gtg` (DDOM/GTG), `vsd_genbo` (VSD/GenBO), `nfbo` (NFBO), `dibo` (DiBO)

## 1. 编写实验JSON

参考已有模板（如 `genbo_tfm_ei_fkl.json`），格式：

```json
{
  "description": "实验描述",
  "experiments": [
    {
      "name": "myalgo_taskA_seed0",
      "paper": "My Paper",
      "suite": "full",
      "cwd": "/data/xk/zhaoyang/算法目录",
      "conda_env": "conda环境名",
      "timeout_sec": 172800,
      "command": "python train.py --task taskA --seed 0 && python eval.py --task taskA --seed 0",
      "env_setup": "export WANDB_MODE=online && export PYTHONPATH=/data/xk/zhaoyang/算法目录:$PYTHONPATH"
    }
  ]
}
```

要点：
- `name` 格式: `算法_task_seed序号`，如 `genbo_TFM-EI-fKL_seqlen15_seed0`
- `conda_env` 别写错，对应关系见上面
- `command` 里 train 和 eval 用 `&&` 串联
- 不要用GPU 0（留给交互式调试）

## 2. 放置到服务器

```bash
scp my_experiments.json xk@180.209.6.83:/data/xk/zhaoyang/repro/
```

## 3. 启动队列

```bash
ssh xk@180.209.6.83

nohup python3 /data/xk/zhaoyang/repro/run_queue.py \
  --experiments /data/xk/zhaoyang/repro/my_experiments.json \
  --suite full \
  --gpus 1,2,3 \
  --max-parallel 3 \
  --skip-completed \
  --logdir /data/xk/zhaoyang/bbo_repro/logs \
  > /data/xk/zhaoyang/bbo_repro/logs/queue_myexp.log 2>&1 &
```

参数说明：
- `--suite full`: 跑完整实验
- `--gpus`: 分配哪些GPU，不包含0
- `--max-parallel`: 并行数 = GPU数
- `--skip-completed`: 自动跳过已完成的（断点续跑）

## 4. 查看进度

```bash
# 看队列状态
tail -f /data/xk/zhaoyang/bbo_repro/logs/queue_myexp.log

# 看GPU使用
nvidia-smi

# 看某个实验的实时输出
tail -f /data/xk/zhaoyang/bbo_repro/logs/full/myexp_seed0.gpu1.log
```

## 5. 提取结果

### DDOM / GTG
从日志中grep:
```bash
grep "best score" /data/xk/zhaoyang/bbo_repro/logs/full/ddom_full_*.log
grep "nmax_ep_reward" /data/xk/zhaoyang/GTG-main/outputs.log
```

使用归一化:
```python
norm_y = (raw_y - y_min) / (y_max - y_min)
# 参考 bounds: DKitty[-880.5, 340.9], Ant[-386.9, 590.2], 
#              Superconductor[0, 74], TFBind8[0, 1], TFBind10[-1.9, 2.1]
# 注意: DDOM和GTG的superconductor bounds不同！
```

### GenBO / VSD
在wandb云端查看，或从日志目录提取。

## 6. 停止/重跑

```bash
# 杀队列
kill $(ps aux | grep "run_queue.py" | grep -v grep | awk '{print $2}')

# 删掉失败的日志让 --skip-completed 失效
rm /data/xk/zhaoyang/bbo_repro/logs/full/bad_exp_seed*.log

# 重新启动（用同样的命令）
```

## 常见问题

- **`python` not found**: 用 `python3`
- **队列重复启动**: 先 `kill` 旧进程再启动新的
- **`--skip-completed` 全跳过**: 改 `name` 后缀（如 `_v2`）或删旧日志
- **磁盘满**: 日志和checkpoint在 `/data` 下，及时清理
- **wandb离线**: 检查 `env_setup` 里有没有 `export WANDB_MODE=online`
