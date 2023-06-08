import json

negatedlama = [json.loads(line) for line in open("negatedlama_endingonly.jsonl")]
negatedlama = [
    item
    for item in negatedlama
    if item["metadata"]["dataset"] in ["ConceptNet", "TREx"]
]

obqa = [json.loads(line) for line in open("obqa.jsonl")]
obqa = [item for item in obqa if item["negation_rule"] in ["not/be", "not/because"]]

all_data = negatedlama + obqa

for i, q in enumerate(all_data):
    # Answer = B if even
    if i % 2 == 0 and q["answerKey"] == "A":
        q["question"]["choices"] = q["question"]["choices"][::-1]
        q["answerKey"] = "B"
    # Answer = A if odd
    if i % 2 == 1 and q["answerKey"] == "B":
        q["question"]["choices"] = q["question"]["choices"][::-1]
        q["answerKey"] = "A"

    q["question"]["choices"][0]["label"] = "A"
    q["question"]["choices"][1]["label"] = "B"

with open("final_submission.jsonl", "w") as f:
    for item in all_data:
        f.write(json.dumps(item) + "\n")
