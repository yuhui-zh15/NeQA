# Generating Process

1. `negatedlama.jsonl` & `obqa.jsonl` from previous folders

2. `remove_nonending_questions.py`: `negatedlama.jsonl` -> `negatedlama_endingonly.jsonl`

3. `filter.py`: `negatedlama_endingonly.jsonl` + `obqa.jsonl` -> `final_submission.jsonl`

4. `json2csv.py`: `final_submission.jsonl` -> `final_submission.csv.1` + `final_submission.csv.2`
