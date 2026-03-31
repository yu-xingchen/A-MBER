# Mini Benchmark S001-S005 (2026-03-31)

## Scope

This file records the first small benchmark set built from:

- `scenario_001`
- `scenario_002`
- `scenario_003`
- `scenario_004`
- `scenario_005`

The goal of this set is not to be the final benchmark. It is a compact, iterated testbed for checking:

- whether question difficulty is high enough
- whether late-session easy summary questions have been reduced
- whether the benchmark relies more on path/mechanism questions than result-summary questions
- whether the current question curation rules are stable enough to reuse

## Current combined dataset

Combined units file:

- [all_units_s001_s005_pruned_20260331.json](/C:/Users/chen4/Desktop/红熊/emotion_memory_benchmark_package/emotion_memory_benchmark_package_pure_text/data/generated_batches/blueprint_review_5pack_closure_mix/all_units_s001_s005_pruned_20260331.json)

Current total:

- `88` items

## Scenario counts

- `scenario_001`: `19`
- `scenario_002`: `19`
- `scenario_003`: `17`
- `scenario_004`: `17`
- `scenario_005`: `16`

## Combined distribution

### Memory level

- `Level 3`: `65`
- `Level 2`: `4`
- `Level 1`: `12`
- `Level 0`: `7`

### Content type

- `long_term_implicit_emotion`: `51`
- `long_term_explicit_or_semi_explicit_emotion`: `18`
- `long_term_fact`: `9`
- `instant_emotion`: `7`
- `near_term_fact`: `2`
- `relation_change`: `1`

### Question type

- `judgment`: `45`
- `explanation`: `34`
- `retrieval`: `7`
- `modality_ambiguous`: `1`
- `modality_missing`: `1`

### Reasoning structure

- `multi-hop`: `40`
- `trajectory-based`: `17`
- `conflict-resolution`: `14`
- `single-hop`: `13`
- `direct`: `4`

### Anchor session

- `D1`: `8`
- `D2`: `21`
- `D3`: `26`
- `D4`: `22`
- `D5`: `11`

### Adversarial type

- `none`: `77`
- `pseudo_relevant_history`: `9`
- `insufficient_evidence`: `2`

## Current curation rules that matter most

The current mini benchmark is not prompt-only. It depends on post-generation curation.

The most important active rules are:

- same-anchor light deduplication
- late result-over-path question downweighting
- obvious-easy pruning
- hard-core structural filtering
- support count tied to kept hard-core count

Files:

- [question_curation.py](/C:/Users/chen4/Desktop/红熊/emotion_memory_benchmark_package/emotion_memory_benchmark_package_pure_text/question_curation.py)
- [question_pipeline.py](/C:/Users/chen4/Desktop/红熊/emotion_memory_benchmark_package/emotion_memory_benchmark_package_pure_text/question_pipeline.py)
- [question_phase_planning.py](/C:/Users/chen4/Desktop/红熊/emotion_memory_benchmark_package/emotion_memory_benchmark_package_pure_text/question_phase_planning.py)
- [question_generation_runtime.py](/C:/Users/chen4/Desktop/红熊/emotion_memory_benchmark_package/emotion_memory_benchmark_package_pure_text/question_generation_runtime.py)

## What was pruned most recently

The latest cleanup pass mainly removed:

- shallow single-hop retrieval questions
- late-session result-summary questions that were too easy

This particularly affected:

- `scenario_002`
- `scenario_003`

while `scenario_001`, `scenario_004`, and `scenario_005` were already relatively clean under the current rules.

## Intended use

This set should be treated as:

- the first reusable mini benchmark
- the baseline set for short evaluation cycles
- the reference set for future prompt and curation comparisons

It is small enough to iterate on quickly, but already structured enough to expose:

- overreliance on late-session summary cues
- weak retrieval controls
- false hard questions under `session_local`
- differences across closure profiles

## Langfuse baseline

Dataset:

- `emotion-memory-mini-s001-s005-pruned-20260331`

Main run:

- [mini-s001-s005-pruned-session-local-20260331](https://cloud.langfuse.com/project/cmmeku583023wad070qwioswm/datasets/cmne79qq401c1ad071fg88vmp/runs/b9fab974-f59e-45c8-82df-d24ca71b3e62)
- local report: [langfuse_mini_s001_s005_pruned_session_local_20260331.txt](/C:/Users/chen4/Desktop/红熊/emotion_memory_benchmark_package/emotion_memory_benchmark_package_pure_text/docs/langfuse_mini_s001_s005_pruned_session_local_20260331.txt)

Retry run for failed items:

- [mini-s001-s005-pruned-retry-session-local-20260331](https://cloud.langfuse.com/project/cmmeku583023wad070qwioswm/datasets/cmne7w9n401fvad076mi124qs/runs/b65a9850-ec06-430c-b487-cd509f646c7e)
- local report: [langfuse_mini_s001_s005_pruned_retry_session_local_20260331.txt](/C:/Users/chen4/Desktop/红熊/emotion_memory_benchmark_package/emotion_memory_benchmark_package_pure_text/docs/langfuse_mini_s001_s005_pruned_retry_session_local_20260331.txt)

### Main run only

- successful items: `73`
- `llm_judge_final`: `0.707`
- `judge_core_answer_correct`: `0.740`
- `exact_match`: `0.744`
- `judge_key_point_coverage`: `0.635`
- `judge_rationale_alignment`: `0.638`
- `judge_unsupported_claims`: `0.860`
- `judge_insufficiency_handling`: `0.868`

### Retry run only

- retry items: `15`
- `llm_judge_final`: `0.778`
- `judge_core_answer_correct`: `0.807`
- `exact_match`: `0.625`
- `judge_key_point_coverage`: `0.707`
- `judge_rationale_alignment`: `0.664`
- `judge_unsupported_claims`: `1.000`
- `judge_insufficiency_handling`: `1.000`

### Combined 88-item baseline

This combines the main run and retry run at the item level.

- total items: `88`
- open-ended judged items: `43`
- option / exact-match items: `49`
- `llm_judge_final`: `0.720`
- `judge_core_answer_correct`: `0.752`
- `exact_match`: `0.722`
- `judge_key_point_coverage`: `0.649`
- `judge_rationale_alignment`: `0.643`
- `judge_unsupported_claims`: `0.886`
- `judge_insufficiency_handling`: `0.892`

## Claude Sonnet 4.6 comparison

Task model:

- `zhongzhuan_claude_sonnet46`

Judge model:

- `portkey_gemini25_flash`

Reports:

- main: [langfuse_mini_s001_s005_claude46_session_local_20260331.txt](/C:/Users/chen4/Desktop/红熊/emotion_memory_benchmark_package/emotion_memory_benchmark_package_pure_text/docs/langfuse_mini_s001_s005_claude46_session_local_20260331.txt)
- retry 1: [langfuse_mini_s001_s005_claude46_retry_session_local_20260331.txt](/C:/Users/chen4/Desktop/红熊/emotion_memory_benchmark_package/emotion_memory_benchmark_package_pure_text/docs/langfuse_mini_s001_s005_claude46_retry_session_local_20260331.txt)
- retry 2: [langfuse_mini_s001_s005_claude46_retry2_session_local_20260331.txt](/C:/Users/chen4/Desktop/红熊/emotion_memory_benchmark_package/emotion_memory_benchmark_package_pure_text/docs/langfuse_mini_s001_s005_claude46_retry2_session_local_20260331.txt)
- retry 3: [langfuse_mini_s001_s005_claude46_retry3b_session_local_20260331.txt](/C:/Users/chen4/Desktop/红熊/emotion_memory_benchmark_package/emotion_memory_benchmark_package_pure_text/docs/langfuse_mini_s001_s005_claude46_retry3b_session_local_20260331.txt)

### Combined 88-item Claude result

- total items: `88`
- open-ended judged items: `44`
- option / exact-match items: `51`
- `llm_judge_final`: `0.617`
- `judge_core_answer_correct`: `0.600`
- `exact_match`: `0.613`
- `judge_key_point_coverage`: `0.525`
- `judge_rationale_alignment`: `0.528`
- `judge_unsupported_claims`: `0.961`
- `judge_insufficiency_handling`: `0.974`

### Gemini vs Claude (current mini benchmark)

- Gemini baseline is stronger on this set.
- Claude Sonnet 4.6 is much closer to the target `60%-70%` band.
- The gap is especially visible on:
  - `llm_judge_final`: `0.720 -> 0.617`
  - `judge_core_answer_correct`: `0.752 -> 0.600`
  - `exact_match`: `0.722 -> 0.613`
  - `judge_key_point_coverage`: `0.649 -> 0.525`
