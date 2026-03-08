# HumanLM 项目理解与设计想法

## 1. 我对这个项目的整体判断

这个 `humanlm/` 不是一个从零重写的独立框架，而是构建在 `verl` 之上的一个 recipe。它复用了 `verl` 的 PPO/GRPO 训练器、rollout 基础设施和 reward manager 机制，再用少量但关键的定制，把训练目标从“模仿用户回复文本”改成“对齐用户潜在心理状态”。

如果和同仓库里的 `collabllm/` 对比，可以更容易看清它的定位：

- `collabllm` 优化的是“首轮回答能不能带来更好的后续多轮协作”。
- `humanlm` 优化的是“当前用户生成是否反映了这个人的 stance / emotion / belief / value / goal / communication / response”。

所以 HumanLM 的核心变化不是换了一个 trainer，而是换了训练单位。它不再把一条 user response 当作唯一目标，而是把它拆成多个 state-alignment 子问题。

## 2. 论文思想到当前代码的映射

### 2.1 状态空间

论文的核心主张是：真实用户回复并不只是 surface text imitation，而是由一组更稳定、心理上更有解释力的 latent states 驱动。这里在代码中被具体化为 `humanlm/state_config/*.json`。

默认 RL 配置 `sebvgcr.json` 对应 7 个层级：

- `stance`
- `emotion`
- `belief`
- `value`
- `goal`
- `communication`
- `response`

这些 state 不是隐藏向量，而是自然语言字段。这个选择很重要，因为它让 reward、debug、人工检查都变得可解释。

### 2.2 数据构造

`humanlm/process_dataset.py` 做了两件事：

1. 把 Humanual 数据集转成 `verl` 需要的 parquet 格式。
2. 把 persona、context、ground-truth response 打包成 SFT/RL 样本。

一个很关键的观察是：RL 数据里并没有 ground-truth state label。`reward_model.ground_truth` 只保存真实 response。也就是说，这个项目不是在做“监督式状态标注学习”，而是在做“让模型生成的 state 能被 judge 认为足够解释 ground-truth response”。

这和论文的 ad-hoc state alignment 非常一致：state 的监督信号来自 response，而不是来自大规模人工 state annotation。

### 2.3 训练时如何把一个样本变成多个状态任务

`humanlm/state_dataset.py` 是整个项目最关键的实现之一。

它在训练时把同一条原始样本复制成多个版本，每个版本只负责一个 `state_name`，并替换成该 state 专属的 system prompt。这样做的效果是：

- 同一个 persona + context，会被要求分别生成 `belief`、`emotion`、`goal` 等不同字段。
- 同一个 policy 被训练成一个多任务 user simulator，而不是单纯 response generator。

这一步非常贴近论文里“先学 aligned latent states，再利用它们生成 response”的思想，但当前实现仍然是一个近似版：它是把多个 state 当作并列训练任务，而不是显式地先采样一整组 states，再条件化生成 response。

### 2.4 rollout 与 reward

`humanlm/humanlm_agent_loop.py` 负责 rollout，`humanlm/reward_function.py` 负责打分。

这里的流程可以概括成：

1. `StateDataset` 产出某个 state-specific prompt。
2. `HumanLMAgentLoop` 用 chat template 渲染 prompt，并按 state 生成内容。
3. 对同一个 `(global_step, index, state_name)` 采样 4 个 rollout。
4. `HumanLMRewardManager` 把这 4 个 rollout 聚到一起。
5. `metrics/state_reward.py` 用 LLM-as-judge 根据 ground-truth response 给 state 对齐打分。
6. reward 只落在当前生成的最后一个有效 token 上，由 GRPO 更新。

这套实现很干净，因为它没有引入复杂的多阶段 actor，只是在 `verl` 现有接口内把“sample expansion + grouped reward + state-conditioned prompt”拼起来了。

## 3. 我认为最关键的深层洞察

### 3.1 当前代码实现的是“可解释的任务分解”，还不是“显式的状态变量模型”

这是我看完论文和代码后最重要的判断。

论文叙事上更像：

- 先生成 aligned latent states
- 再把这些 aligned states synthesis 成 response

但当前代码更像：

- 用同一个 policy 分别学习多个 state-specific generation task
- 再把 `response` 也当作其中一个 task 一起训练

也就是说，`response` 目前并不是显式条件化在模型刚刚生成出的 `belief/value/goal/...` 上，而只是另一个共享参数的 sibling task。这种做法有两个好处：

- 工程上简单，稳定，容易复用 `verl`
- 不需要显式构造 state-to-response 的生成链

但它也有明显代价：

- state 之间的联合一致性没有被强约束
- response synthesis 这一步没有真正显式化
- 训练到最后，模型更像“会分别回答很多 state 问题”，而不是“先形成状态、再统一说话的人”

所以如果后面要做“更像论文 full method”的功能，我认为最值得补的就是显式 synthesis。

### 3.2 这个项目的 reward，本质上是在学习“哪些状态描述足以解释真实回复”

这里的监督来源非常微妙。

因为没有人工标 state，模型生成的 `belief`、`emotion`、`goal` 等字段，是通过 judge 看它们是否能解释 ground-truth response 来得到奖励。这意味着它学到的不是唯一真实 state，而是“对这个 response 最有解释力、最可接受的一组 state 描述”。

这带来两个后果：

- 好处：不需要昂贵 state annotation，容易扩展到多数据域。
- 风险：同一个 response 可能对应多组 equally plausible states，reward 会有不可辨识性。

换句话说，当前 reward 更像是在优化 post-hoc explanatory states，而不是保证恢复出用户心里唯一真实的 latent state。这并不是缺陷，而是 HumanLM 能规模化落地的关键前提。但如果以后要加新功能，最好承认这个建模前提，而不是假设这些状态一定“客观真实”。

### 3.3 strict format 在早期训练里可能比 state semantics 更强

`reward_function.py` 中的 `strict_format=True` 加上短字段、固定 XML tag、state-specific stop sequence，会让模型先学会“格式合规”。

这对训练稳定性是有帮助的，因为 judge 不需要处理脏输出。但这也意味着：

- valid rate 可能在前期主导 reward
- 模型可能先学 tag compliance，再学 state semantics
- state reward 里的一部分 improvement，未必都是语义对齐提升，也可能只是解析成功率提升

这提示一个后续优化方向：可以把训练拆成 curriculum。

例如：

- 第一阶段宽松格式，只学 state content
- 第二阶段再提高 strict parsing / exact tag 要求

或者至少把 format reward 和 semantic reward 分开记账，不然训练曲线容易误读。

### 3.4 `separate_generation=True` 提高了稳定性，但切断了 state 间依赖

现在的训练策略本质上是“一个 state 一个 rollout”。这很好，因为：

- 搜索空间小
- judge 更容易打分
- 每个 state 的 max_tokens 很短，样本效率高

但代价也很明显：

- `belief` 和 `value` 之间可能互相矛盾
- `goal` 和 `communication` 之间可能风格不一致
- `response` 没有被显式约束去使用前面已经生成出来的 states

也就是说，当前实现强调的是 per-state correctness，而不是 cross-state coherence。

如果后面真要做更强的人类模拟，这个 coherence gap 迟早要补。

### 3.5 persona dropout 是一个值得放大的好点子

`process_dataset.py` 里对 `test_dropout` split 的 persona 字段做 dropout，这件事我觉得非常有价值。它说明这个项目已经意识到真实部署时一个核心难点：persona 永远不完整。

我认为这条线很值得继续往下做，因为真实 user simulation 的难度，不只是 state generation，而是：

- persona 缺失
- 上下文稀疏
- 用户状态会随对话动态变化

如果以后要扩功能，我会优先把“partial persona robustness”做成一个正式训练目标，而不只是测试扰动。

## 4. 当前实现里我会特别留意的工程问题

这些不一定是致命 bug，但会直接影响后续开发体验。

### 4.1 SFT 脚本和 HumanLM 命名还没有完全收口

`humanlm/train_sft_humanlm.sh` 里还有明显的旧路径和旧命名残留，比如：

- `recipe/usim/...`
- `trainer.project_name=usim`
- 一些硬编码数据路径

这说明 SFT 部分有一层历史包袱。后面如果你要在 SFT 上继续扩展状态生成或 synthesis，建议先把这层命名和路径清干净，不然很容易出现“能跑但语义已经不一致”的问题。

### 4.2 no-repeat ngram 配置现在看起来没有真正接上

`train_rl_humanlm.sh` 里虽然定义了 `NO_REPEAT_NGRAM_SIZE`，但实际传给 trainer 的是：

- `+actor_rollout_ref.no_repeat_ngram_size=0`

而不是脚本变量。

另外 `humanlm_agent_loop.py` 里触发 no-repeat 时会 import `recipe.humanlm.no_repeat_ngram`，但当前类其实定义在同文件里。也就是说，这条分支大概率没有被认真走通过。

这类问题不会影响主路径训练，因为默认就是 0；但如果后面你要在 eval 或 response mode 上控制重复，它会马上变成坑。

### 4.3 `eval_state_name` 的设计意图和实际行为可能不一致

`StateDataset` 里有 `eval_state_name` 配置，但在 `__getitem__` 的非训练路径中又把 `extra_info.state_name` 强行改成了 `response`。

这意味着“验证时只评某个非 response state”的能力，至少从当前实现上看并不干净。这个点如果后续要做更细粒度 ablation，会值得先修。

### 4.4 路径命名还带着 `recipe.humanlm` 的历史结构假设

多个文件里都写的是 `recipe.humanlm.*` 路径，但你当前工作目录里真实存在的是 `humanlm/`。这说明代码假设的包结构和你眼下仓库结构之间存在一层历史映射。

短期内也许还能跑，但如果后面要做更系统的模块化、单测或者迁移，很建议先统一 import/path 约定。

## 5. 我认为最值得做的扩展方向

如果让我选，我会按下面的优先级推进。

### 方向 A：补上显式 state-to-response synthesis

这是最贴近论文完整方法的一步，也是我认为价值最大的功能增强。

目标不是再把 `response` 当作一个独立 sibling task，而是改成：

1. 先生成一组 states
2. 再让 response 显式条件化在这些 generated states 上
3. 对 response 再加一个 alignment reward

可以有两种实现路线：

- 两阶段 rollout：先生成 states，再把 states 拼进第二个 prompt 生成 response
- 单次 structured rollout：一次生成 `think + states + response`

我更倾向第一种。虽然更重，但 credit assignment 更清楚，也更符合论文叙事。

### 方向 B：加入 cross-state consistency reward

如果暂时不想重构成两阶段生成，那至少应该补一个 consistency 层。

例如：

- `belief` 是否支持 `stance`
- `value` 是否支持 `goal`
- `communication` 是否和最终 `response` 风格一致

这类 reward 不需要人工标注，可以继续走 LLM judge，但评估对象从单字段改成字段对/字段组。这样能弥补 `separate_generation=True` 的天然缺口。

### 方向 C：把 persona dropout 从测试扰动升级为训练正则

现在 persona dropout 主要用于 `test_dropout`。我认为更好的方式是：

- 训练时随机 mask 一部分 persona 字段
- 让模型学会在不完整 persona 下仍然维持 state consistency
- 同时在 reward 里区分“依赖 persona 的 state”和“依赖 context 的 state”

这样会更接近真实使用场景。

### 方向 D：蒸馏 LLM judge，降低 reward 成本和方差

当前 `state_reward` / `state_reward_on_response` 很明显是训练成本大户，而且 judge 方差一定不低。

一个现实可行的方向是：

1. 先继续用现有 judge 收集训练日志
2. 积累 `(context, response, generated_state, judge_score)` 数据
3. 针对每个 state 训轻量 reward model
4. 用蒸馏 reward 替代大模型 judge 或做 hybrid judge

这会让实验迭代快很多。

## 6. 如果我接下来在这个目录里继续做功能，我会怎么切入口

我会把改动入口分成四层：

- 状态定义层：`state_config/*.json`
- prompt/render 层：`system_prompts/*.txt`、`state_dataset.py`、`humanlm_agent_loop.py`
- reward 层：`reward_function.py`、`metrics/*.py`
- 数据层：`process_dataset.py`

如果你的新功能是“新增一种 state / 改 reward / 改生成顺序 / 引入显式 synthesis”，这四层基本就是全部需要动的地方。

## 7. 一句话总结

我目前对这个项目的判断是：

HumanLM 已经把论文里最有价值的部分落地了，也就是“用可解释 state alignment 替代纯 response imitation”。但当前代码实现更像一个非常聪明、工程上可落地的近似版本，而不是论文 full pipeline 的完全显式化版本。它最强的地方是 state decomposition + grouped GRPO + LLM judge；它最值得继续补强的地方，是 state 之间的联合一致性，以及真正显式的 state-to-response synthesis。
