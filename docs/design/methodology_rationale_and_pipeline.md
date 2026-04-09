# Long-Horizon Affective Memory Benchmark: Design Rationale and End-to-End Workflow

## 1. Scope and design target

This benchmark is designed to evaluate whether a model can use long-horizon conversational memory to interpret a user's current emotional state, with a primary emphasis on **implicit emotion**. The benchmark is not intended to be a broad-spectrum emotion dataset or a generic long-context QA dataset. Instead, it targets a narrower but more diagnostic question: when the current utterance is not sufficient on its own, can a model retrieve and use historically relevant interaction evidence to explain the user's present affective state?

The benchmark is instantiated in a **psychological teacher–student** scenario. This setting is chosen because it simultaneously satisfies several practical and methodological requirements:

1. **Stable role structure.** The interaction naturally fits an agent–user configuration, which aligns with RedBear AI's intended application pattern.
2. **Emotion-rich but controllable context.** Compared with open-ended casual chat, a psychological teacher–student setting naturally supports recurring stressors, misunderstandings, support, disappointment, defensiveness, and partial repair.
3. **Clear evaluation target.** The benchmark evaluates only the student's emotional state, reducing label ambiguity and preventing symmetric two-sided emotion modeling from diluting the main task.
4. **Longitudinal realism.** The scenario plausibly supports multiple sessions, repeated topics, cumulative events, and relation-state fluctuations without becoming overly dramatic.

Accordingly, the benchmark focuses on the student's emotional state as the single target variable, while the teacher/agent is treated as a support and interaction partner rather than a co-equal affective target.

## 2. Three-layer task design rationale

The benchmark uses a **core–diagnostic–stress-test** structure.

### 2.1 Core layer
The core layer defines the benchmark's main distribution. It contains:
- long-term implicit emotion judgment
- long-term implicit emotion retrieval
- long-term implicit emotion explanation
- long-term explicit / semi-explicit emotion items
- long-term and near-term factual control items
- instant emotion control items
- conflict-resolution items under text–audio mismatch

The core layer exists to answer the main research question: does memory help with affective interpretation, especially when the current turn is not self-sufficient?

### 2.2 Diagnostic layer
The diagnostic layer is smaller and exists to identify *why* a model succeeds or fails on the core layer. It includes:
- relation-state judgment
- relation-change judgment
- trajectory-based emotion items
- multi-hop factual retrieval / explanation

These items do not redefine the benchmark's center of gravity. Their role is analytic: they decompose long-horizon affective understanding into interpretable sub-capabilities such as relation modeling, trajectory modeling, and multi-hop history integration.

### 2.3 Stress-test layer
The stress-test layer probes robustness under non-ideal input conditions. It includes:
- modality-missing items
- modality-ambiguous items
- comparison items
- ranking items
- open explanation items

These tasks are intentionally smaller in quantity because they are designed to stress the model rather than define the main data distribution. They are important because real deployment settings frequently involve incomplete, noisy, or weak signals.

## 3. Adversarial design rationale

Adversarial items are included as a **horizontal tag**, not as a separate top-level task family. Their goal is not to artificially increase difficulty, but to assess:

- resistance to **pseudo-relevant history**
- calibration under **insufficient evidence**
- resistance to **over-interpretation** of vague or weak affective signals
- robustness to **pseudo-conflict** between modalities

In affective tasks, a model can easily drift into a harmful pattern: treating every vague, polite, or short response as a hidden negative emotion. Adversarial items prevent this by requiring the model either to select the truly relevant dialogue evidence or to withhold overconfident affective conclusions when evidence is insufficient.

## 4. Why evidence must be dialogue-grounded

A crucial design decision is that benchmark evidence must be grounded in **dialogue turns**, not only in hidden event scripts.

Internal event scripts remain useful for construction:
- planning longitudinal arcs
- enforcing consistency
- tracking critical events
- generating distractors

However, the benchmark model only sees:
- current and historical dialogue turns
- text and modality metadata
- optionally audio features or audio files

Therefore, gold evidence used for evaluation must point to **observable dialogue units**. This leads to a dual evidence structure:

- **Internal construction layer:** `critical_event_ids`
- **Benchmark-facing evidence layer:** `evidence_turn_ids`, optionally `evidence_turn_quotes`

This design aligns annotation, evaluation, and model input, and prevents hidden script-level information from leaking into the benchmark's evidence standard.

## 5. Recommended MVP

A strong MVP is:
- 5 sessions
- 150 turns total
- 3 stages
- 4–6 implicit emotion states
- text + audio
- approximately 60 derived questions

A suitable three-stage arc is:
1. initial support building
2. misunderstanding and relational fluctuation
3. accumulated disappointment with partial repair

Suitable implicit states include:
- disappointment
- defensiveness
- grievance
- distancing
- suppressed helplessness

## 6. Data objects to generate before dialogue generation

Before generating the full dialogue, prepare the following structured objects:

1. `personas.json`
2. `global_outline.json`
3. `session_scripts.json`
4. `event_plan.json`
5. `emotion_arc.json`
6. `question_plan.json`

The benchmark should not begin with unconstrained dialogue generation. Instead, it should begin with a controlled backbone that later constrains the conversation.

## 7. End-to-end workflow

### Step 1: write personas
Create:
- student persona
- teacher persona

Each persona should focus on emotionally relevant traits, not broad fictional detail. For the student, include pressure sources, sensitivity points, expression style, and coping pattern. For the teacher, include support style, strengths, and blind spots.

### Step 2: define the global outline
Create a three-stage structure that specifies:
- stage goals
- relation states
- dominant emotional background
- stage function in the long-horizon arc

### Step 3: write session scripts
For each session, specify:
- theme
- goal
- turn count
- major events
- dominant student emotions
- relation state
- measurable points
- future memories to plant
- constraints

This step is critical because it prevents the conversation from drifting while leaving room for natural surface realization.

### Step 4: create the event plan
The event plan should record:
- event ID
- event type
- session
- description
- emotional significance
- relation impact
- whether the event can be a critical memory
- whether it can be a distractor

This table supports long-term factual items, retrieval items, ranking items, and adversarial distractor design.

### Step 5: define the emotion arc
Map the student's emotional progression across stages:
- dominant stage-level emotions
- implicit states to seed
- relation states
- stage-to-stage changes

This keeps the generated dialogue emotionally coherent across sessions.

### Step 6: generate the conversation in structured JSON
Use the dialogue generation prompt to create a LoCoMo-style conversation object. Every dialogue turn should include:
- `dia_id`
- `turn_index_global`
- `speaker`
- `text`
- `audio_id` (optional; only when a real audio reference exists)
- `voice_style` (required delivery description used when no audio file is available)
- `modality_available`

The conversation object is the canonical benchmark input surface.

### Step 7: annotate turn-level emotion states
Use the annotation prompt to produce student-focused turn-level annotations. Each annotation should include:
- underlying emotion
- implicit vs explicit
- expression style
- relation state
- memory dependency level
- reasoning structure
- critical event IDs
- evidence turn IDs
- gold rationale

All evidence used for annotation must be grounded in dialogue turn IDs.

### Step 8: generate questions
Use the question generation prompt to derive:
- judgment items
- retrieval items
- explanation items
- modality-missing items
- modality-ambiguous items
- comparison items
- ranking items
- open-generation items
- adversarial items

Each question should include:
- anchor dialogue ID
- content type
- question type
- memory level
- reasoning structure
- evidence turn IDs
- gold rationale

### Step 9: run quality control
Perform at least four checks:
1. **Arc consistency:** the student emotion and relation trajectories should not drift illogically.
2. **Evidence observability:** every benchmark-facing rationale must be explainable from turn-level evidence.
3. **Task validity:** a Level 3 item must genuinely require long-horizon history.
4. **Calibration control:** not every vague turn should be labeled as a hidden negative emotion.

### Step 10: run adversarial checks
Verify that adversarial items are interpretable and fair:
- pseudo-relevant distractors should be plausible, not random
- insufficient-evidence items should be truly underdetermined
- pseudo-conflict items should not artificially manufacture impossible ambiguity

## 8. Suggested file organization

A practical package may contain:

- `personas.json`
- `global_outline.json`
- `session_scripts.json`
- `event_plan.json`
- `emotion_arc.json`
- `question_plan.json`
- `conversation.json`
- `annotation.json`
- `qa.json`
- `all_units.json`

This design supports a two-layer representation:
- **interaction-level files** for generation and planning
- **question-level files** for evaluation

## 9. Recommended paper framing

A useful way to describe the benchmark in a paper is:

- The **core layer** defines the primary task distribution centered on long-horizon implicit emotion understanding.
- The **diagnostic layer** provides interpretable probes into relation modeling, trajectory modeling, and complex evidence integration.
- The **stress-test layer** evaluates robustness under missing or ambiguous modality conditions.
- **Adversarial items** measure resistance to pseudo-relevant evidence and overconfident inference under insufficient support.

This framing helps justify why auxiliary and enhanced task families are smaller in quantity: their purpose is not to dilute the benchmark's center, but to expose mechanism-level strengths and weaknesses.

## 10. Reusable implementation principles

1. Generate structured plans before free-form dialogue.
2. Treat event scripts as internal scaffolds, not final evaluation evidence.
3. Ground all benchmark evidence in turn-level dialogue IDs.
4. Keep the evaluation target fixed to one side of the interaction.
5. Use adversarial design to test calibration, not just task difficulty.
6. Prefer a smaller but coherent stress-test set over a large but noisy one.

## Appendix A. Implementation Checklist

### A. Planning
- [ ] Write `personas.json`
- [ ] Write `global_outline.json`
- [ ] Write `session_scripts.json`
- [ ] Write `event_plan.json`
- [ ] Write `emotion_arc.json`
- [ ] Compute `question_plan.json`

### B. Generation
- [ ] Run `dialogue_generation_prompt_v2.j2`
- [ ] Validate JSON structure against `conversation.schema.json`
- [ ] Confirm every turn has `dia_id`, `turn_index_global`, `text`, and modality metadata

### C. Annotation
- [ ] Run `annotation_prompt_v2.j2`
- [ ] Validate against `annotation.schema.json`
- [ ] Confirm every annotation item contains `evidence_turn_ids`
- [ ] Confirm no item uses only hidden events as evidence

### D. Question generation
- [ ] Run `question_generation_prompt_v2.j2`
- [ ] Validate against `qa.schema.json`
- [ ] Confirm every question uses `anchor_dia_id` and `evidence_turn_ids`

### E. Quality control
- [ ] Check stage arc consistency
- [ ] Check relation-state consistency
- [ ] Check Level 3 items really require long-horizon history
- [ ] Check adversarial items are fair and interpretable
- [ ] Check modality-missing / modality-ambiguous items are realistic
