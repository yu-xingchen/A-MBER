# Benchmark Iteration Log (2026-03-31)

## Scope

This note records the concrete iteration path that led from the earlier prompt-heavy QA generation setup to the current `scenario_002` checkpoint that scores around the target difficulty band in Langfuse.

It is intentionally narrower than the methodology document:
- this file explains what we changed and why
- `methodology_rationale_and_pipeline.md` explains the broader benchmark design
- `langfuse_eval_workflow_zh.md` explains how to run evaluation

## Goal of this iteration

The original issue was not that the pipeline could not run. The issue was that generated questions were too easy, especially because:

- late-session explicit summary turns were overused
- result-summary questions were overrepresented
- too many questions could be answered from a single session
- question generation kept drifting toward safe anchors even after prompt tightening

The target was to make the benchmark harder and more diagnostic without turning every scenario into the same unresolved template.

## Major changes

## 1. Blueprint: closure diversity and benchmark-oriented arcs

We changed blueprint generation so that closure is no longer assumed to be a single full-repair arc.

Main changes:
- added a closure profile pool
- made `global_outline.closure_profile` explicit
- moved closure-profile selection from prompt-only to pipeline-controlled selection
- used light persona-aware weighting instead of stressor-level hardcoding

Closure profiles currently used:
- `partial_repair_with_caution`
- `practical_progress_relational_residue`
- `clearer_relational_improvement_one_unresolved_thread`
- `mixed_or_stalled_progress`
- `surface_closure_hidden_residue`
- `recalibrated_working_alliance`

Files:
- [scenario_blueprint_structure_generation_prompt_v1.j2](../prompts/scenario_blueprint_structure_generation_prompt_v1.j2)
- [scenario_blueprint_detail_generation_prompt_v1.j2](../prompts/scenario_blueprint_detail_generation_prompt_v1.j2)
- [pipeline.py](../pipeline.py)
- [global_outline.schema.json](../schemas/global_outline.schema.json)

Why this mattered:
- it reduced closure collapse
- it made earlier and middle sessions more useful for QA
- it prevented every scenario from defaulting to the same 鈥減artial relief with residue鈥?ending

## 2. Dialogue: benchmark-first instead of realism-first

We shifted dialogue generation away from 鈥渕ust sound realistic and counseling-like鈥?toward 鈥渕ust generate useful benchmark evidence.鈥?
Main changes:
- fewer explicit outcome-summary lines
- more observable behavioral changes
- better closure-profile realization
- stronger emphasis on local shifts and benchmark-worthy residue
- voice_style restricted toward observable delivery

Files:
- [dialogue_generation_prompt_v2.j2](../prompts/dialogue_generation_prompt_v2.j2)

We also compressed the dialogue prompt to reduce latency and prompt bloat.

## 3. Conversation robustness: sanitizer and timeout support

Conversation generation often failed for two reasons:
- API timeout
- voice_style validation drift

To stabilize this:
- conversation request timeout was increased
- raw conversation saving was preserved
- local normalization was expanded so small voice_style violations can be repaired without rerunning the whole model call

Files:
- [pipeline.py](../pipeline.py)

Examples of stabilization:
- broader `VOICE_STYLE_SANITIZE_REPLACEMENTS`
- saving `conversation_raw.json` and validation errors
- normalized local rescue path for otherwise-usable conversations

## 4. Annotation: acceptable, but not the main bottleneck

Annotation was reviewed and found to be good enough to continue the pipeline.

Observed behavior:
- it did not collapse entirely to late-session summary turns
- it surfaced good D2/D3/D4 anchors
- it still occasionally included some late explicit anchors

Conclusion:
- annotation was not the current main bottleneck
- the more important problem was how question generation used those anchors

## 5. Question generation: from prompt-only control to structural control

This was the biggest shift.

Earlier iterations only tightened prompts:
- fewer explicit-summary anchors
- stronger path/mechanism preference
- more cross-session requirements

Those changes helped, but the model still found safe solutions.

We then moved to structural control:
- two-phase generation
- anchor-first hard-core generation
- structural hard-core postprocessing
- oversample-then-prune
- hard-count drives support-count
- same-anchor light deduplication

### 5.1 Two-phase generation

Question generation was split into:
- `hard_core`
- `everything_else`

Purpose:
- separate genuinely hard mechanism/reinterpretation questions from support/control questions

### 5.2 Anchor-first hard-core

Instead of letting the model freely choose anchors for hardest questions, we:
- scored annotations
- built a preferred anchor pool
- downweighted late explicit summary anchors
- passed preferred/avoid anchor hints into the phase plan

### 5.3 Structural hard-core postprocessing

We deliberately kept this narrow and generalizable.

Current hard-core filtering rules mainly target:
- single-session pseudo-hard items
- late explicit summary anchors
- shallow retrieval items

Important design choice:
- code handles structural thresholds
- prompt handles softer semantic preference

### 5.4 Oversample then prune

Instead of generating exactly the target count, the pipeline now oversamples each phase and prunes later.

Current behavior:
- target 12 hard -> generate 15
- target 12 support -> generate 15

This gives the pruning stage actual choice instead of forcing weak questions to remain.

### 5.5 Hard drives support

We changed the final count logic:
- support count no longer needs to stay fixed
- if only a smaller number of hard questions survive quality thresholds, support shrinks with it

This matches the benchmark goal better:
- strong hard questions define the core
- support items should surround the core, not inflate it artificially

### 5.6 Same-anchor light deduplication

We added a light final-stage deduplication rule:
- if two questions share the same `anchor_dia_id`, keep the one that looks more benchmark-central
- prefer higher-memory, non-retrieval, more diagnostic questions

This is intentionally light:
- it removes obvious duplicate-anchor waste
- it avoids deep semantic hardcoding

Files:
- [question_generation_prompt_v2.j2](../prompts/question_generation_prompt_v2.j2)
- [pipeline.py](../pipeline.py)
- [question_curation.py](../question_curation.py)

## Current `scenario_002` checkpoint

Key checkpoint files:
- current QA: [qa.json](../data/generated_batches/blueprint_review_5pack_closure_mix/scenario_002/qa.json)
- pre-hard-drives-support QA: [qa_pre_hard_drives_support_20260331.json](../data/generated_batches/blueprint_review_5pack_closure_mix/scenario_002/qa_pre_hard_drives_support_20260331.json)
- local anchor-dedup output: [qa_local_anchor_dedup_20260331.json](../data/generated_batches/blueprint_review_5pack_closure_mix/scenario_002/qa_local_anchor_dedup_20260331.json)

Observed improvement in the latest main QA:
- fewer `D5` anchors than earlier versions
- more `D2` / `D3` / `D4` anchors
- much higher `Level 3` share
- much lower `Level 1` share

Representative strong anchors now include:
- `D3:24`
- `D4:18`
- `D2:28`
- `D2:14`
- `D3:20`
- `D4:2`
- `D3:12`

## Langfuse result for the current `scenario_002` checkpoint

Run: `79b2e3da` (blueprint-review-s002-session-local-20260331)

Local report:
- [langfuse_blueprint_review_s002_session_local_20260331.txt](../docs/langfuse_blueprint_review_s002_session_local_20260331.txt)

Average scores:
- `llm_judge_final`: `0.575`
- `judge_core_answer_correct`: `0.609`
- `exact_match`: `0.615`
- `judge_key_point_coverage`: `0.486`
- `judge_rationale_alignment`: `0.482`
- `judge_unsupported_claims`: `0.818`
- `judge_insufficiency_handling`: `0.727`

Interpretation:
- the benchmark now sits much closer to the intended 60% difficulty band
- failures are concentrated more in explanation quality than in pure answer extraction
- this suggests the benchmark is increasingly testing path reasoning, not just result recognition

## Code organization note

`pipeline.py` had grown too large.

As part of this iteration, the QA curation logic was partially extracted into:
- [question_curation.py](../question_curation.py)

This is not a full refactor yet, but it is the first step toward making the question-generation portion easier to maintain.

## Recommended next step

Do not immediately add more prompt restrictions.

The higher-value next move is:
- run the current pipeline on another scenario with a different closure profile
- verify that the new question-generation structure generalizes

Recommended target:
- a scenario that is not `surface_closure_hidden_residue`

## Suggested document merge opportunities

Likely worth keeping separate:
- `methodology_rationale_and_pipeline.md`
- `langfuse_eval_workflow_zh.md`
- this iteration log

Likely worth merging or consolidating later:
- `implementation_checklist.md`
  - could become a short appendix section inside the methodology document
- checkpoint `.txt` notes such as:
  - `question_generation_checkpoint_20260330_185042_notes.txt`
  - similar checkpoint notes saved beside generated artifacts
  - these can eventually be summarized into this iteration log instead of staying as separate loose notes

Not recommended to merge into one giant file:
- large Langfuse `.txt` raw reports
- workflow instructions
- methodology rationale
- iteration notes

Those serve different purposes and are easier to use when separate.

