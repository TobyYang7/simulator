# HumanLM Training Flow

这个文档按当前 `recipe/humanlm` 目录里的训练脚本和实现代码，记录 HumanLM 的训练流程。

## 1. 训练入口

- RL 入口：`train_rl_humanlm.sh`
- SFT 入口：`train_sft_humanlm.sh`
- 数据预处理：`process_dataset.py`
- RL 自定义数据集：`state_dataset.py`
- RL agent loop：`humanlm_agent_loop.py`
- RL reward：`reward_function.py`

HumanLM 的核心目标不是直接模仿“用户回复文本”，而是让模型学习生成和用户心理状态一致的内容结构。

## 2. 数据准备

### 2.1 `process_dataset.py` 的职责

`process_dataset.py` 负责把原始 Humanual 数据整理成两种格式：

- SFT 格式
- RL 格式

其中 RL 样本会保留：

- `prompt`
- `reward_model`
- `extra_info`
- `data_source`

`extra_info` 里会带上很多后续训练要用的元信息，例如：

- `index`
- `name`
- `state_name`
- `persona`
- `post`
- `target_user_id`

这些字段后面会被 system prompt 模板和奖励函数使用。

### 2.2 State 配置

HumanLM 的关键是“按状态层级训练”。状态定义来自 `state_config/*.json`，例如：

- `sebvgcr.json`
- `r.json`
- `r_no_tag.json`

这些配置决定：

- 要训练哪些状态字段
- 每个状态对应的 system prompt 模板
- 训练时要求模型输出哪些 XML 风格 tag，例如 `<belief>...</belief>`

`train_humanlm` 模式默认使用：

- `sebvgcr.json`

也就是一个多状态层级配置，而不只是最终 `response`。

## 3. RL 脚本做了什么

`train_rl_humanlm.sh` 会先读取：

- `cluster_config.sh`
- 其中指定的 `.env`

用来拿模型路径、数据目录、缓存目录、输出目录等环境配置。

### 3.1 运行模式

脚本主要有两种模式：

- `train_humanlm`
- `eval_only`

其中：

- `train_humanlm` 用于真正训练 HumanLM
- `eval_only` 用于只做验证/评估，不继续训练

### 3.2 `train_humanlm` 模式的关键设置

在 `train_humanlm` 下，脚本会默认启用：

- `CONFIG=sebvgcr`
- `ENABLE_THINKING=True`
- `ENABLE_HETERO_THINK=True`
- `MAX_GEN_LENGTH=1024`
- `SEPARATE_GENERATION=True`
- `USE_DIFF_H_SYS_PROMPTS=True`
- `ENABLE_STATE=False`

这意味着训练时模型会：

- 使用带 thinking 的模板
- 按不同状态分别生成
- 为不同状态使用不同 system prompt

### 3.3 启动的训练器

RL 入口同样是：

```bash
python3 -m verl.trainer.main_ppo
```

但这里也是按：

- `algorithm.adv_estimator=grpo`

来跑 GRPO 风格训练。

## 4. RL 数据在训练时如何被展开

### 4.1 `StateDataset` 的作用

RL 不是直接用默认的 `verl` 数据集类，而是显式指定：

- `data.custom_cls.name='StateDataset'`

`StateDataset` 在普通 RLHF 数据集之上加了两件关键事：

1. 可以按 state config 替换 system prompt。
2. 可以把同一条原始样本扩展成多个“状态视角”的训练样本。

当 `augment_with_states=True` 时，同一个样本会被复制成多个版本，每个版本对应一个 `state_name`，例如：

- `stance`
- `emotion`
- `belief`
- `value`
- `goal`
- `communication`
- `response`

每个版本会套用对应状态的 system prompt，让模型只生成该状态应该输出的部分。

### 4.2 为什么这样做

HumanLM 的训练单位不再是“整条用户回复文本”，而是：

- 对某个状态字段的受控生成

这样 reward 就能针对状态维度做对齐，而不是只看最终表面文本像不像。

## 5. Rollout 阶段怎么生成

### 5.1 自定义 agent loop

脚本把默认 agent loop 指向：

- `HumanLMAgentLoop`

并通过 `humanlm_agent_loop_config.yaml` 注册它。

### 5.2 `HumanLMAgentLoop` 的行为

`HumanLMAgentLoop.run()` 做的事情比较直接：

1. 读取 `raw_prompt`
2. 读取 `extra_info.state_name`
3. 根据当前状态决定 chat template 的渲染方式
4. 如果开启 `enable_hetero_think`，则在非 `response` 状态下关闭 thinking
5. 调用 tokenizer 的 `apply_chat_template`
6. 用 vLLM 生成一段文本

这里和 `collabllm` 最大的区别是：

- HumanLM 不做“未来多轮对话模拟”
- 它的 rollout 更像是“按当前状态生成一个结构化字段”

### 5.3 停止条件

脚本会根据 `state_config` 自动推导 stop sequence，例如：

- `</belief>`
- `</emotion>`
- `</response>`

这保证模型在生成完当前状态对应的 tag 后就停下。

## 6. Reward 是怎么工作的

### 6.1 reward 入口

脚本里同时配置了：

- `reward.reward_manager.name=HumanLMRewardManager`
- `custom_reward_function.name=compute_reward`

两者都来自 `reward_function.py`。

其中：

- `compute_reward` 负责按 metric 批量算分
- `HumanLMRewardManager` 负责把 rollout 样本聚合、解析、加权、写回 reward tensor

### 6.2 训练时的多 rollout

脚本设置：

- `reward_model.reward_kwargs.n_rollouts=4`
- `actor_rollout_ref.rollout.n=4`

也就是同一个 prompt/state 在训练时会采样 4 次生成。

`HumanLMRewardManager` 会用一个稳定 key 把同一步、同一个样本、同一个 state 的 4 次 rollout 聚到一起，然后统一打分。这个 key 由以下信息组成：

- `global_step`
- `state_name`
- `index`

### 6.3 先解析模型输出

模型输出后，reward manager 会先把文本解析成字段。

如果开启 `strict_format`，它会严格要求输出满足类似这种格式：

```text
<belief>...</belief>
<emotion>...</emotion>
...
```

如果格式不合法，就会被判成无效，影响 valid rate 和 reward。

在 `SEPARATE_GENERATION=True` 的训练模式下，每个样本通常只要求生成当前 `state_name` 对应的字段，而不是一次生成全部字段。

### 6.4 metric 计算

reward manager 会按 `field -> metric` 的配置去调用 `metrics/*.py`。

默认训练配置里，核心 metric 是：

- `state_reward`

它的作用是评估当前生成内容是否和目标状态描述对齐。

验证时还可能额外使用：

- `state_reward_on_response`

用于在最终回复上做更细的状态评估。

### 6.5 聚合方式

对同一个样本的 4 次 rollout，reward manager 会：

1. 解析出各次生成的目标字段
2. 对重复内容去重后再调用 metric，减少重复开销
3. 把每个字段、每个 metric 的得分映射回原始 rollout
4. 乘上 `field_metric_weights`
5. 把所有字段和 metric 相加，得到单个 rollout 的总分

### 6.6 reward 落点

和 `collabllm` 一样，最终 reward 会被写到：

- 当前 response 的最后一个有效 token

所以 PPO/GRPO 更新的仍然是本次生成，而不是某个逐 token 的 dense reward。

## 7. `train_humanlm` 模式可以怎么理解

如果按默认的 `train_humanlm` 模式理解，完整流程是：

1. 从处理好的 RL parquet 读取样本。
2. `StateDataset` 按状态配置把样本扩展成多个 state-specific 训练项。
3. `HumanLMAgentLoop` 针对某个 state 生成带 tag 的结构化内容。
4. 同一个样本/state 采样 4 次 rollout。
5. `HumanLMRewardManager` 把这 4 次结果聚在一起，检查格式并计算状态对齐奖励。
6. 把总 reward 打到本次生成末尾 token。
7. `verl` 用 GRPO 更新模型。

本质上，HumanLM 不是在优化“回复像不像人”，而是在优化：

- 模型生成的各个心理状态字段，是否和目标 persona / 场景 / 用户状态一致。

## 8. SFT 流程

`train_sft_humanlm.sh` 是监督微调入口。

它会：

1. 读取处理好的 `train.parquet` / `val.parquet`
2. 调用 `verl.trainer.fsdp_sft_trainer`
3. 用 `prompt -> generation` 做常规 SFT
4. 根据 `thinking` 或 `no_thinking` 选择不同数据目录和训练模板

SFT 的目标是先把模型调到一个合理的初始分布；RL 再在这个基础上强化“状态一致性”。

## 9. 和 CollabLLM 的主要区别

CollabLLM 强调：

- 首轮回答是否能带来更好的后续协作对话

HumanLM 强调：

- 当前生成内容是否和目标用户状态对齐

所以两者虽然都用 `verl + GRPO`，但优化目标不同：

- CollabLLM 是“未来多轮交互质量”
- HumanLM 是“结构化状态对齐质量”
