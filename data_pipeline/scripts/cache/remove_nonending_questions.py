import json

data = [json.loads(line) for line in open("final_submission/cache/negatedlama.jsonl")]
data = [item for item in data if "___" not in item["question"]["stem"]]

with open("final_submission/cache/negatedlama_endingonly.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")