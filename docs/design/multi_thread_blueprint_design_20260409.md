# Multi-Thread Blueprint Design

## Goal
Move scenario blueprints away from a single-problem arc and toward 2-3 concurrent life threads that create mixed emotions, interference, and cross-thread reinterpretation.

## Core decision
Do not build one flat pool of `event + fixed emotion labels`.

Use:
- event threads for "what is happening"
- appraisal pressure for "why this matters emotionally"
- session weaving for "which thread is foregrounded, which one leaks in, and how they interfere"

This keeps the benchmark grounded in realistic student behavior rather than explicit emotion labeling.

## Thread model
Each scenario should usually contain:
- 1 primary thread
- 1 secondary thread
- 1 lower-volume background thread

Example thread families:
- academic evaluation or performance pressure
- friendship or romance uncertainty
- family request, money, autonomy, or permission conflict
- workload and time-fragmentation pressure
- identity, belonging, or comparison pressure

## Why threads instead of fixed emotion cards
If the blueprint directly samples emotion labels such as `happy`, `worried`, or `ashamed`, the scenario tends to become:
- too explicit
- too repetitive across runs
- too easy for local sentiment reading

Thread-based planning lets mixed emotions emerge naturally:
- romance can create excitement plus worry
- presentation pressure can amplify perfectionism and irritability
- family-request conflict can create hesitation, shame, and strategic politeness

## Operational rule
Within the existing schema, represent multi-thread structure through:
- `global_outline.overall_arc`
- `stages[*].goal`
- `stages[*].emotional_background`
- `session_scripts[*].session_theme`
- `session_scripts[*].main_topics`
- `session_scripts[*].major_events`
- `session_scripts[*].future_memory_to_plant`

No schema expansion is required for the first experiment.

## Should thread count be a config?
Yes, but only as a soft control.

Recommended behavior:
- keep `default_active_thread_count = 3`
- allow `desired_thread_count` as an optional config later
- treat it as a target, not a strict guarantee for every session

Why not make it too rigid:
- if enforced too hard, session plans start to read like inventory lists
- some personas or closure shapes support 3 strong threads better than 4

Recommended future config shape:
- `desired_thread_count`: default `3`
- `background_noise_density`: low / medium
- `cross_thread_interference_min`: default `2`

For now, prompt-level enforcement is enough.

## Blueprint design rules
Each scenario should satisfy:
- 2-3 live threads across the full scenario
- at least 2 cross-thread interference moments
- at least 1 thread that stays partly unresolved by the end
- mixed student emotions in at least 4 sessions
- no single confession carrying the entire arc

## Session weave rule
Each session should usually contain:
- 1 foregrounded thread
- 1 secondary thread leak or interference
- 1 relationship movement

This prevents the conversation from becoming a list of unrelated problems while still increasing emotional complexity.

## Event-writing rule
Prefer event descriptions that show:
- one thread being talked about directly while another thread shapes the student's tone or constraints
- selective disclosure
- softened disagreement
- face-saving agreement
- practical language carrying emotional weight

Avoid:
- session plans that are just "topic A, then topic B, then topic C"
- explicit lists of feelings without interactional evidence

## Emotional design rule
Use short affective states, but let them be mixed.

Good combinations:
- hopeful + worried
- relieved + embarrassed
- excited + guilty
- validated + cautious
- overloaded + determined

Avoid relying on only one dominant emotion per session when the thread structure supports more complexity.

## Noise and chitchat
Natural conversations need some irrelevant or low-relevance material.

Recommended use:
- 1-2 low-stakes mentions in many sessions
- some should appear only once
- they should feel normal rather than suspiciously benchmark-designed

Good noise examples:
- room change
- printer issue
- shuttle delay
- coffee or food aside
- weather comment
- battery or charger issue
- roommate noise
- a minor social scheduling annoyance

What noise is for:
- making transcripts feel lived-in
- creating distractors and local clutter
- preventing every turn from sounding benchmark-optimal

What noise is not for:
- replacing real thread structure
- becoming a hidden fourth major arc accidentally
- flooding the conversation with meaningless filler

## Exploratory parameter choice
For the first blueprint-only experiment, use:
- `sessions_per_conversation = 7`
- `turns_per_conversation = 168`
- `stage_count = 6`
- `fixed_events_per_session = 4`

Why:
- 7 sessions is enough to support 2-3 ongoing threads without every session feeling overcrowded.
- 168 turns leaves room for repeated motifs and cross-thread interference.
- 6 stages allows the arc to change shape without forcing a neat linear progression.
- 4 events per session is enough to support one foreground event, one interference event, one relation movement, and one later-memory hook.

## What to inspect in the first blueprint batch
We are not looking for polished final scenarios yet.
We are checking whether the generated blueprints show:
- multiple active stressor families
- mixed emotions rather than single-label sessions
- thread interference rather than simple topic rotation
- early and middle sessions that already contain benchmark-worthy ambiguity
- endings that do not collapse into one neat closure template
