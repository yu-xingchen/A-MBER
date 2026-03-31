import json, sys
from collections import Counter

qa = json.load(open(sys.argv[1], encoding='utf-8'))
print(f"Total questions: {len(qa)}")
print()

types = Counter(q['question_type'] for q in qa)
print("Question types:", dict(types))

levels = Counter(q['memory_level'] for q in qa)
print("Memory levels:", dict(levels))

reasoning = Counter(q['reasoning_structure'] for q in qa)
print("Reasoning:", dict(reasoning))

adv = [q for q in qa if q.get('adversarial_flag')]
print(f"\nAdversarial: {len(adv)}")
for a in adv:
    qid = a["question_id"]
    atype = a["adversarial_type"]
    text = a["question_text"][:80]
    print(f"  {qid}: {atype} - {text}")

modality = Counter(q.get('modality_condition', 'normal') for q in qa)
print(f"\nModality conditions: {dict(modality)}")

print("\nCross-session evidence analysis:")
for q in qa:
    evidence = q.get('evidence_turn_ids', [])
    sessions = set()
    for e in evidence:
        if ':' in e:
            sessions.add(e.split(':')[0])
    qid = q["question_id"]
    qt = q["question_type"]
    ml = q["memory_level"]
    rs = q["reasoning_structure"]
    print(f"  {qid} ({qt}, {ml}, {rs}): {len(sessions)} sessions, {len(evidence)} turns")
