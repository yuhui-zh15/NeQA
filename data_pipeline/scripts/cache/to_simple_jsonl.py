import json

original_data = [json.loads(line) for line in open("all.jsonl")]
processed_data = []

for i, q in enumerate(original_data):
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


for item in original_data:
    assert len(item["question"]["choices"]) == 2
    assert item["question"]["choices"][0]["label"] == "A"
    assert item["question"]["choices"][1]["label"] == "B"

    if "metadata" in item:
        metadata = item.get("metadata", None)
    else:
        metadata = {
            "id": item["id"],
            "split": item["split"],
            "negation_rule": item["negation_rule"],
        }
    processed_data.append(
        {
            "question": item["question"]["stem"],
            "choices": [
                item["question"]["choices"][0]["text"],
                item["question"]["choices"][1]["text"],
            ],
            "answer": {"A": 0, "B": 1}[item["answerKey"]],
            "metadata": metadata,
        }
    )

with open("all_simple.jsonl", "w") as f:
    for item in processed_data:
        f.write(json.dumps(item) + "\n")
