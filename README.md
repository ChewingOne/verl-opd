# verl-opd
在 verl 上实现 OPD（On-Policy Distillation）

## OPD 实现说明

本文档说明本仓库为支持 OPD 训练模式所做的改动、实现原理，以及如何运行与测试。

## 修改过程

本次实现按以下顺序完成：

1. 仓库侦察与入口确认
- PPO 训练入口与主编排：`verl/trainer/main_ppo.py`、`verl/trainer/ppo/ray_trainer.py`
- rollout 统一接口：`verl/workers/rollout/base.py`（`generate_sequences`）
- 现有 PPO/KL/loss 位置：`verl/trainer/ppo/core_algos.py`、`verl/workers/utils/losses.py`
- worker 与分布式机制：Ray（`Role` + `RayWorkerGroup`）

2. 新增 OPD 配置
- `verl/trainer/config/algorithm/opd.yaml`
- `verl/trainer/config/opd_trainer.yaml`

3. 新增 OPD 训练实现
- `verl/trainer/opd/trainer.py`
- `verl/trainer/opd/loss.py`
- `verl/trainer/opd/__init__.py`

4. 新增 Teacher 推理 worker
- `verl/workers/ref/teacher_logprob_worker.py`
- `verl/workers/ref/__init__.py`

5. 新增启动入口
- 直接 OPD 入口：`verl/trainer/main_opd.py`
- 统一分发入口：`verl/trainer/main.py`（按 `algorithm.name` 分发）

6. 新增测试
- `tests/trainer/opd/test_loss.py`

## 实现原理

### 1. 训练范式（On-Policy）

每个训练步执行：

1. 读取一批 prompts（大小 `B`）
2. 用当前 student 策略对每个 prompt 采样 `K` 条 completion
3. teacher 对同一轨迹（`prompt + 生成 token`）做前向
4. 计算 token-level reverse KL：`KL(pi_student || pi_teacher)`
5. 仅在 completion token 上计算 loss（同时屏蔽 padding）
6. 更新 student；下一步重新采样，保持 on-policy

### 2. Reverse KL 定义

默认使用 full-vocab teacher logits：

- `logp_s = log_softmax(student_logits[:, :-1, :])`
- `logp_t = log_softmax(teacher_logits[:, :-1, :])`
- `p_s = exp(logp_s)`
- `kl_token = sum_v p_s(v) * (logp_s(v) - logp_t(v))`

然后按 mask 聚合为 `mean` 或 `sum`。

### 3. Shift 对齐与 Mask 规则

next-token 对齐规则：

- `logits[:, :-1, :]` 预测 `input_ids[:, 1:]`

loss mask 形状为 `[B*K, T-1]`，并满足：

- padding 位置不参与 loss
- `mask_prompt_tokens=true` 时，prompt 区间不参与 loss
- completion 的 shift 后索引区间为：`[prompt_len-1, prompt_len-1+gen_len)`

### 4. Teacher top-k 近似（可选）

- `approx_teacher_topk=0`：默认 full vocab（推荐）
- `approx_teacher_topk>0`：启用 top-k 近似，会改变 KL 目标并引入偏差

## 关键改动文件

- `verl/trainer/config/algorithm/opd.yaml`
- `verl/trainer/config/opd_trainer.yaml`
- `verl/trainer/main.py`
- `verl/trainer/main_opd.py`
- `verl/trainer/opd/__init__.py`
- `verl/trainer/opd/loss.py`
- `verl/trainer/opd/trainer.py`
- `verl/workers/ref/__init__.py`
- `verl/workers/ref/teacher_logprob_worker.py`
- `tests/trainer/opd/test_loss.py`

## 如何执行

### 1. 环境准备

请先安装项目依赖，至少包括：

- `torch`
- `transformers`
- `hydra-core`

### 2. 直接运行 OPD

```bash
python -m verl.trainer.main_opd \
  data.train_files=/path/to/train.parquet \
  model.path=/path/to/student_model \
  algorithm.teacher_model_path=/path/to/teacher_model \
  algorithm.num_samples_per_prompt=2 \
  algorithm.max_new_tokens=256
```

### 3. 通过统一入口运行

```bash
python -m verl.trainer.main --config-name opd_trainer \
  model.path=/path/to/student_model \
  algorithm.teacher_model_path=/path/to/teacher_model \
  algorithm.name=opd
```

### 4. 常用配置项

- `algorithm.num_samples_per_prompt`：每个 prompt 的采样数 `K`
- `algorithm.max_new_tokens`：completion 最大长度
- `algorithm.temperature` / `algorithm.top_p` / `algorithm.top_k`：student 采样参数
- `algorithm.mask_prompt_tokens`：是否仅在 completion token 上计算 loss
- `algorithm.kl_reduction`：`mean` 或 `sum`
- `algorithm.approx_teacher_topk`：teacher 近似开关（默认 `0`）
- `algorithm.batch_size_prompts`：每步 prompt 批大小 `B`
- `algorithm.microbatch_size`：前向 micro-batch
- `algorithm.gradient_accumulation`：梯度累积步数

## 如何测试

运行 OPD 单测：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/trainer/opd/test_loss.py -q
```

覆盖内容：

1. shift 对齐正确性
2. mask 规则正确性（仅 completion，且不含 padding）
3. teacher==student 时 KL 接近 0
4. tiny 优化回路中 KL 呈下降趋势

## 已知注意事项

- 若环境缺少 `torch` 或 `hydra-core`，训练与测试会失败
- `approx_teacher_topk>0` 为近似目标，不等价于 full-vocab KL
