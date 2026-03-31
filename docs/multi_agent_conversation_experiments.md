# Multi-Agent Conversation Experiments

This note records the first round of multi-agent conversation experiments for the benchmark construction pipeline. The goal of this branch of work is not to replace the current single-agent workflow immediately, but to test whether a teacher-student split can improve persona realization without breaking interaction-blueprint fidelity or downstream task-layer quality.

The experiments were run on the same interaction blueprint as `scenario_003`, using a copied scenario directory `scenario_003_m` so that the only intended difference was the conversation generator. Earlier generated conversations were renamed as `v1`, `v2`, and `v3`; the latest run is treated as `v4`. In all cases, the interaction blueprint remained the same and only the conversation realization strategy changed.

## Motivation

The benchmark is built from a staged dependency chain: interaction blueprint, conversation, annotation, task layer, and final benchmark units. In this pipeline, the conversation is not just an intermediate artifact. It determines whether later annotation can recover the intended emotional trajectory, and whether the task layer can still measure the intended long-horizon emotional memory skills. A multi-agent generator is therefore only useful if it preserves the benchmark-relevant structure of the interaction blueprint while still producing natural dialogue.

For this reason, the correct standard is not exact sentence matching against the single-agent version. Some realization-level variation is acceptable and even desirable. What is not acceptable is drift that changes persona logic, event function, or the measurement signal needed for the downstream task layer.

## Experimental Setup

The multi-agent version uses an AutoGen-based split between a `TeacherAgent` and a `StudentAgent`, with a Python controller in `pipeline.py` orchestrating turn order, session boundaries, dialogue metadata, and prompt payloads. The main single-agent workflow remains unchanged; the multi-agent path is only activated through `conversation_generation.mode = "multi_agent"` in the config.

Each turn is generated separately. The controller passes local context for the current session, the currently active event, recent history, and optional retry feedback. Generated turns are normalized into the standard conversation schema so that the rest of the pipeline can continue unchanged.

## Version History

### v1: Free Role Split

The first AutoGen version only separated the teacher and student personas and let the controller alternate turns with very limited steering. This established that a role split can produce fluent dialogue, but blueprint fidelity was poor. The agents often produced a reasonable conversation that was no longer the same scenario. In `scenario_003_m`, this showed up as support actions and subplot choices that were not in the original event plan, especially in later sessions.

This version demonstrated the core failure mode of naive multi-agent generation for this benchmark: the output may remain coherent in isolation while no longer supporting the intended emotional-memory questions.

### An Intermediate Strong-Constraint Direction

During the iteration process, we also explored a stricter controller design that would push the dialogue much more aggressively toward the intended event realization. The idea behind this direction was to make the controller explicitly guard the current interaction beat, so that multi-agent generation could not easily drift into a different but still locally coherent narrative.

That direction remains important conceptually and should be recorded, because it clarified an essential tradeoff in this benchmark: stronger control can improve blueprint fidelity, but it also makes the system less general and more maintenance-heavy. In practice, this style of controller depended too much on event-specific logic. As soon as the event family changed, the controller logic would need to be rewritten or expanded again. For that reason, this strong-constraint direction was not retained as the main path, and the currently preserved `v1`-`v4` conversation files should not be read as a clean one-to-one mapping to that abandoned branch.

### v2-v3: Gradual Soft-Steering Refinement

The middle preserved versions (`v2` and `v3`) are better understood as a gradual shift away from unconstrained role splitting and toward soft steering. Instead of treating every missed beat as a failed turn, the controller began inspecting short recent windows and injecting gentle feedback into the next prompt when the interaction seemed to be drifting. This made the dialogue much more natural than a hard-enforcement design and reduced the most obvious storyline drift seen in `v1`.

However, this logic was still tied directly to `event_type` branches inside the controller. That meant the system could only generalize as far as the current event taxonomy generalized. These versions were useful transition points, but they were not yet principled representations of interaction constraints.

### v4: Blueprint-Like Soft Constraints

The latest version kept the soft-steering idea but changed the internal representation. Instead of driving the controller directly from event-type-specific conditions, the controller now derives a lightweight `interaction_targets` object for each event at runtime. These targets include teacher-side intent, student-side shift, soft signal cues, and anchoring topics. The controller then uses those targets, rather than raw event names, when deciding whether to steer the next turn.

This version is still not fully general, because the `interaction_targets` are derived by code from the current event taxonomy. But it is architecturally closer to a future design in which the blueprint itself would carry those constraints as explicit metadata.

## Observed Results

Across `v1` to `v4`, the main trend is clear. Multi-agent generation can produce readable dialogue, but without additional control it tends to write a different reasonable scene rather than realize the original blueprint faithfully. Later revisions improved this substantially. By `v4`, the conversation is much closer to the intended session-level structure, especially in `S2` and `S3`, and the worst forms of story drift are reduced.

The strongest improvement in `v4` is that the interaction is now recognizably implementing the same high-level scenario rather than a loosely related alternative. Clara’s gradual withdrawal in `S2` is more visible than in earlier multi-agent runs, and Mr. Hayes’s emotional acknowledgement in `S3` is now present early enough to support the intended repair arc. The encoding noise that appeared in earlier runs has also been removed through deterministic text cleanup in the conversation normalization step.

At the same time, `v4` still does not match the single-agent baseline in blueprint fidelity. The remaining weakness is no longer total narrative drift; it is subtler. The revised support plan in `S3` still tends to expand into a new support thread rather than staying fully anchored to the already planted support items in the blueprint. This matters because even if the new support move is reasonable, it can still alter what later QA will treat as the key memory-bearing evidence.

## Interpretation

These experiments support a narrow conclusion and do not yet support a broad one. The narrow conclusion is that the current multi-agent implementation is improving but still trails the single-agent generator in blueprint fidelity. The broad claim that “multi-agent is worse for this task” would not be justified, because the current comparison is still between a mature single-agent workflow and an evolving multi-agent controller.

What these experiments do show is that, for this benchmark, conversation generation must preserve three layers of fidelity if downstream QA is to remain reliable. The first is persona fidelity: the teacher and student must still sound like the intended roles. The second is event fidelity: the key interaction beats must occur in roughly the intended order and function. The third is measurement fidelity: the resulting conversation must still support the intended annotation and QA structure. Realization-level variation is acceptable; drift that breaks any of these three layers is not.

## Why This Matters For QA

For this benchmark, conversation drift is not a cosmetic issue. If the conversation changes the emotional logic of the scenario, the annotation will adapt to the new logic and the QA will follow it. That means the generated questions may still be internally valid while no longer testing the originally intended long-horizon emotional-memory signal.

In practice, the most damaging forms of drift are missing or weakened misattunement, premature repair, persona-inconsistent support moves, and newly introduced support subplots. Any of these can flatten an originally long-term emotional-memory problem into a simpler local inference problem. This is why conversation generation cannot be evaluated only on fluency or naturalness; it must also be evaluated on whether it protects the blueprint’s measurement signal.

## Current Assessment Of v4

The latest version is the first multi-agent run that is plausibly comparable to the single-agent version at the conversation level. It is still weaker than the single-agent baseline, but the gap is now narrow enough to make further comparison meaningful. In earlier runs, the problem was that the multi-agent output had become a different scenario. In `v4`, the problem is mostly limited to residual support-plan drift and a somewhat softened `S2` misattunement.

This is a meaningful change in status. It means the multi-agent branch has moved from “prototype that clearly drifts” to “prototype that can now be evaluated as an alternative realization strategy.”

## A Second Comparison Case: `scenario_001`

To reduce the risk that the observations above were overly specific to `scenario_003_m`, we also ran a controlled comparison on `data/generated_batches/demo_batch_20260326_134041/scenario_001`, duplicating the same blueprint into a single-agent branch and a multi-agent branch and then generating `conversation`, `annotation`, `qa`, and `all_units` separately.

This case sharpened the difference between the two approaches. The multi-agent conversation was more convincing at the level of persona realization. Leo sounded more like a precision-seeking, ambiguity-sensitive student, and his turn-level manner of speaking was often more characteristic than in the single-agent version. However, the single-agent branch still preserved the benchmark-relevant interaction structure more reliably. In particular, the `S2` misattunement and `S3` repair were translated into QA more cleanly in the single-agent path.

The resulting QA files make the distinction clearer than the conversations alone. In the single-agent branch, the stronger questions were tightly centered on the intended sequence of `misattunement -> withdrawal -> repair`, which is exactly the type of cross-session emotional-memory structure the benchmark is supposed to test. In the multi-agent branch, the questions were still valid, but they were more likely to concentrate on Leo's internal self-doubt, rumination, and guardedness as stand-alone states, rather than on how those states were shaped by Ms. Reed's earlier procedural misattunement and later repair. In other words, the multi-agent path preserved persona signals well, but the single-agent path preserved measurement signals better.

This second case therefore strengthens the current practical conclusion: multi-agent generation is beginning to show an advantage in role enactment, but single-agent generation remains the safer default when the main criterion is downstream QA quality and benchmark fidelity.

## Current Limitations

The present controller is still not generalizable enough to be treated as a reusable multi-agent conversation framework. Although the steering logic now uses blueprint-like `interaction_targets`, those targets are still derived internally from the current event taxonomy. In other words, the controller still contains hidden task knowledge. As long as that remains true, the approach will generalize only as far as the existing event vocabulary generalizes.

Another limitation is that the current soft constraints are optimized around this benchmark’s current scenario family. They are suitable for validating whether multi-agent generation is worth pursuing further here, but they should not be mistaken for a finished abstraction.

## Next Steps

The next stage should not be more controller-specific patching. Instead, the system should move toward making the interaction constraints part of the blueprint representation itself. The most promising direction is to make each event or beat optionally carry an interaction-level contract, such as the intended teacher move, the intended student shift, and what should not happen yet. The controller would then become a thin executor of those contracts rather than an interpreter of event labels.

Before making that schema change, one more validation pass is useful. The current `v4` controller should be tested on a few more scenarios from the same batch to verify that the improvement is not specific to `scenario_003_m`. If the pattern holds, then it becomes worthwhile to formalize the new constraint representation in the blueprint files.

In parallel, any future single-agent versus multi-agent comparison should use a fixed set of blueprints and judge the outputs on blueprint fidelity, persona consistency, dialogue naturalness, and downstream QA support. The purpose of that comparison is not to decide which paradigm is universally better, but to determine which one is currently better suited for benchmark construction under this project’s constraints.

## Practical Recommendation

At the current stage of development, the single-agent conversation generator should remain the default path for final benchmark production. The multi-agent path is valuable as an experimental branch and is now good enough to justify further controlled evaluation, but it is not yet stable enough to replace the main workflow.
