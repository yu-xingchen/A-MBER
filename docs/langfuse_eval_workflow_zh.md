# Langfuse 评测工作流

这套工作流先只支持两种评测设置：

- `session_local`：只给模型当前 session 到 anchor 为止的上下文
- `full_history`：给模型从第一段 session 到 anchor 为止的完整历史

暂时不把 RAG 做成正式流程。等检索算法定下来以后，再决定 `retrieval_candidates` 应该怎么切窗和上传。

## 1. 环境变量

运行前需要准备下面这些环境变量：

- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_HOST`，如果不填就走 SDK 默认 host
- `GATEWAY_API_KEY`
- `GATEWAY_BASE_URL`，默认是 `https://api.portkey.ai/v1`
- `GATEWAY_MODEL`，可选，不传就用脚本默认值

## 2. 上传 dataset

先把本地的 `all_units.json` 上传到 Langfuse dataset。

```powershell
python scripts/langfuse_upload_dataset.py `
  --units-path data/generated_batches/demo_batch_20260323_152935/scenario_001/all_units.json `
  --dataset-name emotion-memory-scenario001 `
  --description "Scenario 001 benchmark units"
```

上传后，每个 dataset item 会包含：

- `input`
  - `question_text`
  - `options`
  - `anchor`
  - `benchmark_views.session_local_context`
  - `benchmark_views.full_history_context`
  - `benchmark_views.modality_conditioned_views`
- `expected_output`
  - `gold_answer`
  - `acceptable_answers`
  - `gold_rationale`
- `metadata`
  - `content_type`
  - `question_type`
  - `memory_level`
  - `reasoning_structure`
  - `modality_condition`

## 3. 在 Langfuse 创建 prompt

建议在 Langfuse UI 里创建一个 `chat` prompt，并把它标成 `production`。

这个 prompt 建议使用下面这些变量：

- `{{context_transcript}}`
- `{{question_text}}`
- `{{options_block}}`
- `{{context_policy}}`

推荐的最小 prompt 结构如下。

### system

```text
You are answering a benchmark question about a counseling-style conversation.
Use only the provided context.
If the context is insufficient, say so directly.
If options are provided, answer using the option content, not just the letter.
Return only the final answer text, without extra explanation.
```

### user

```text
Context policy: {{context_policy}}

Conversation context:
{{context_transcript}}

Question:
{{question_text}}

Options:
{{options_block}}
```

如果后面你想让模型返回 JSON，也可以改成：

```text
Return JSON with one field: {"answer": "..."}.
```

当前脚本会优先尝试解析 JSON 里的 `answer`，如果不是 JSON，就直接把整段文本当答案。

## 4. 运行 `session_local` baseline

```powershell
python scripts/langfuse_run_experiment.py `
  --dataset-name emotion-memory-scenario001 `
  --prompt-name emotion-memory-baseline `
  --prompt-label production `
  --context-policy session_local `
  --experiment-name emotion-memory-eval `
  --run-name session-local-v1
```

## 5. 运行 `full_history` memory

```powershell
python scripts/langfuse_run_experiment.py `
  --dataset-name emotion-memory-scenario001 `
  --prompt-name emotion-memory-baseline `
  --prompt-label production `
  --context-policy full_history `
  --experiment-name emotion-memory-eval `
  --run-name full-history-v1
```

## 6. 现在脚本怎么处理模态题

如果某个 item 带 `modality_conditioned_views`，脚本会优先使用模态视图，而不是普通视图。

当前定义是：

- 只处理 anchor 附近的局部 turn
- 不会删掉整条历史证据链
- `voice_style` 不会被完全删除，而是写成：
  - `unavailable`
  - 或 `unclear`

这更接近“局部模态不可依赖，但还能靠历史证据判断”的测试设定。

## 7. 现在的自动 evaluator

当前实验脚本内置了一个最简单的 evaluator：

- `exact_match`

做法是把模型输出、`gold_answer`、`acceptable_answers` 都做基础规范化后再比较。

这个 evaluator 适合先把流程跑通，但还不够覆盖复杂解释题。后面如果你要更稳的分数，可以再加：

- 规则匹配 evaluator
- LLM judge evaluator
- 多指标 evaluator

## 8. 建议的使用顺序

第一步先只跑：

- `session_local`
- `full_history`

先看这两种上下文策略在同一套 dataset 上的差异。等这条链稳定以后，再继续扩到：

- prompt 版本对比
- 不同模型对比
- 更复杂的 evaluator
- RAG 实验
