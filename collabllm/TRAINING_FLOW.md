# CollabLLM Training Flow

这个文档按当前 `recipe/collabllm` 目录里的训练脚本和实现代码，记录 CollabLLM 的训练流程。

## 1. 训练入口

- RL 入口：`train_rl_collabllm.sh`
- SFT 入口：`train_sft_collabllm.sh`
- 数据预处理：`process_dataset.py`

其中核心训练逻辑在 RL 脚本里。它调用：

```bash
python3 -m verl.trainer.main_ppo
```

也就是用 `verl` 的 PPO/GRPO 训练器做 RL。

## 2. 数据准备

RL 训练依赖 `process_dataset.py` 先把原始 Hugging Face 多轮数据转成 `verl` 需要的 parquet。

### 2.1 RL 数据会被整理成这些字段

- `prompt`: 原始多轮消息，前面额外插入一个强调“主动、澄清、交互性”的 system prompt
- `raw_prompt`: 保存原始 prompt，给 rollout 阶段直接使用
- `ground_truth`: 优先取 `single_turn_completion`，否则回退到 `completion`
- `extra_info`: 保留样本元信息，并补充 `interaction_kwargs`
- `reward_model`: `{"style": "rule", "ground_truth": ground_truth}`
- `data_source`: 固定为 `collabllm`
- `agent_name`: 固定为 `collabllm_agent`

### 2.2 `interaction_kwargs` 的作用

`process_dataset.py` 会把下面这些信息塞进 `extra_info["interaction_kwargs"]`：

- `name=collabllm`
- `single_turn_prompt`
- `task_desc`

这部分会在后面的“用户模拟器”里使用，用来生成未来对话。

## 3. RL 主训练流程

`train_rl_collabllm.sh` 的核心配置可以概括为：

- 算法：`algorithm.adv_estimator=grpo`
- 模型：`Qwen/Qwen2.5-7B-Instruct`
- 奖励管理器：`reward_model.reward_manager=collabllm`
- 自定义奖励函数：`conversation_level_reward_func`
- rollout 引擎：`vllm`
- 开启 `multi_turn`

和普通单轮 PPO 不同，这个 recipe 的思路是：

1. 先让当前 policy 对输入 prompt 生成第一条 assistant 回复。
2. 再基于这条回复，额外模拟几条“未来对话”。
3. 用这些未来对话的质量，反过来给第一条回复打分。
4. PPO/GRPO 只更新“第一条回复”的 token。

## 4. Rollout 阶段是怎么跑的

`config/agent.yaml` 指定默认 agent loop 是 `CollabLLMAgentLoop`。

### 4.1 第一步：先生成首轮 assistant 回复

`CollabLLMAgentLoop.run()` 会先：

- 读取 `raw_prompt`
- 初始化 interaction（如果配了 interaction config）
- 调用 `verl` 的生成逻辑，产出当前模型的第一条回答

这条回答就是训练时真正被当成 `response_ids` 的内容。

### 4.2 第二步：复制状态，采样未来对话

如果第一条回答没有提前终止，agent loop 会把当前状态复制多份：

- `actor_rollout_ref.rollout.multi_turn.num_repeat_rollouts=3`

也就是对同一个样本额外生成 3 条未来对话轨迹。

这些未来轨迹不会直接作为训练目标，而是只用于评分。

### 4.3 第三步：用户模拟器接管“用户”

`config/collabllm_interaction_config.yaml` 配置了一个 interaction：

- 类：`CollabLLMInteraction`
- 用户模型：`gpt-4o-mini`

这个 interaction 不是训练模型本身，而是一个“用户模拟器”。它会读取：

- 任务描述 `task_desc`
- 原始单轮问题 `single_turn_prompt`
- 当前聊天历史

然后让 `gpt-4o-mini` 扮演一个真实用户，生成下一句用户回复。它的设计目标是：

- 尽量像真实用户
- 前期故意不说全
- 让 assistant 通过追问和澄清来推进任务

如果用户模拟器认为任务已经完成，会输出 `[[TERMINATE CHAT]]`，用作停止信号。

### 4.4 未来对话内容会被保存

agent loop 最终把这些未来对话放进：

- `extra_fields["messages"]`

后面的 reward manager 会直接读取这里的多条对话记录。

## 5. 奖励是怎么计算的

### 5.1 奖励入口

`reward_function.py` 注册了 `CollabLLMRewardManager`，名字就是脚本里的：

- `reward_model.reward_manager=collabllm`

它还会调用自定义函数：

- `conversation_level_reward_func`

这个函数会按名字动态加载 `metrics/*.py` 里的 metric 实现。

### 5.2 默认奖励项

RL 脚本默认启用了三项：

- `accuracy=1`
- `interactivity=1`
- `token_amount=-0.0001`

含义分别是：

- `accuracy`: 最终任务答案是否正确
- `interactivity`: assistant 是否理解用户、会澄清、会给建议
- `token_amount`: 未来对话越长，惩罚越大，避免无意义拉长对话

### 5.3 多条未来对话如何聚合

reward manager 会把同一个样本的 3 条未来对话全部取出来：

1. 对每条未来对话分别计算所有 metric。
2. 对同一个 metric，先把 3 次 rollout 的分数求和。
3. 再除以 `num_repeat_rollouts` 得到平均值。
4. 乘上对应权重。
5. 每个 metric 的结果 clamp 到 `[-1, 1]`。
6. 所有 metric 相加，得到该样本总 reward。

### 5.4 reward 落在哪个 token 上

最终 reward 不是分配到整段回复的每个 token，而是只打到：

- 当前这条 assistant 首轮回复的“最后一个有效 token”

因此这个 recipe 实际上是在学习：

- “什么样的首轮回答，能让后续的协作对话更好”

而不是直接监督整段未来对话本身。

## 6. SFT 流程

`train_sft_collabllm.sh` 是一个可选的预训练步骤。

它做的事情比较直接：

1. 读取 `sft_train.parquet` / `sft_validation.parquet`
2. 调用 `verl.trainer.fsdp_sft_trainer`
3. 用多轮 `prompt` 做常规监督微调

这个阶段的作用主要是先把模型调到“更像一个主动协作式助手”的初始状态，再进入 RL。

## 7. 一句话总结

CollabLLM 的训练核心不是“拟合用户回复”，而是：

- 用模型先回答一次
- 用 LLM 用户模拟器继续把对话往后推几轮
- 再用未来对话的正确性、交互性、长度来给这次首轮回答打分
- 最后用 PPO/GRPO 更新这次首轮回答

因此它优化的是“首轮回答对后续协作质量的长期影响”。
