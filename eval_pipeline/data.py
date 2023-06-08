import json
import random


class NeQA:
    def __init__(self, max_instances: int):  # max_instances: -1 for all
        self.path = "neqa_100.jsonl"
        # self.path = "neqa_100_nonnegated.jsonl"
        self.data = [json.loads(line) for line in open(self.path)]
        if max_instances != -1:
            random.seed(42)
            random.shuffle(self.data)
            self.data = self.data[:max_instances]

    def apply(self, prompt_fn: callable):
        for item in self.data:
            item["prompt"] = prompt_fn(item)

    def prompt_zeroshot(self, item: dict) -> str:
        assert len(item["choices"]) == 2
        return f"The following are multiple choice questions (with answers) about common sense.\n\nQuestion: {item['question']}\nA. {item['choices'][0]}\nB. {item['choices'][1]}\nAnswer:"

    def prompt_zeroshot_hint(self, item: dict) -> str:
        assert len(item["choices"]) == 2
        return f"The following are multiple choice questions (with answers) about common sense. Note that if there is a negation in the question, we should choose the wrong answer to the original question.\n\nQuestion: {item['question']}\nA. {item['choices'][0]}\nB. {item['choices'][1]}\nAnswer:"

    def prompt_fewshot(self, item: dict) -> str:
        assert len(item["choices"]) == 2
        return f"The following are multiple choice questions (with answers) about common sense.\n\nQuestion: If a cat has a body temp that is below average, it isn't in\nA. danger\nB. safe ranges\nAnswer: B\n\nQuestion: As the barometer reading goes lower there is not a greater chance of\nA. sunshine\nB. getting wet\nAnswer: A\n\nQuestion: Coral is a type of living organism which cannot be identified in\nA. saltwater locations that are open\nB. any where with liquid\nAnswer: B\n\nQuestion: {item['question']}\nA. {item['choices'][0]}\nB. {item['choices'][1]}\nAnswer:"

    def prompt_fewshot_hint(self, item: dict) -> str:
        assert len(item["choices"]) == 2
        return f"The following are multiple choice questions (with answers) about common sense. Note that if there is a negation in the question, we should choose the wrong answer to the non-negated question.\n\nQuestion: If a cat has a body temp that is below average, it isn't in\nA. danger\nB. safe ranges\nAnswer: B\n\nQuestion: As the barometer reading goes lower there is not a greater chance of\nA. sunshine\nB. getting wet\nAnswer: A\n\nQuestion: Coral is a type of living organism which cannot be identified in\nA. saltwater locations that are open\nB. any where with liquid\nAnswer: B\n\nQuestion: {item['question']}\nA. {item['choices'][0]}\nB. {item['choices'][1]}\nAnswer:"

    def prompt_fewshot_cot(self, item: dict) -> str:
        assert len(item["choices"]) == 2
        return f'The following are multiple choice questions (with answers) about common sense.\n\nQuestion: If a cat has a body temp that is below average, it isn\'t in\nA. danger\nB. safe ranges\nAnswer: Let\'s think step-by-step.\nFirst, let\'s answer non-negated question: "If a cat has a body temp that is below average, it is in?"\nFor this non-negated question, we have A.\nThen, to answer the negated question, take the other answer, which would be B.\nSo the answer is B.\n\nQuestion: As the barometer reading goes lower there is not a greater chance of\nA. sunshine\nB. getting wet\nAnswer: Let\'s think step-by-step.\nFirst, let\'s answer non-negated question: "As the barometer reading goes lower there is a greater chance of?"\nFor this non-negated question, we have B.\nThen, to answer the negated question, take the other answer, which would be A.\nSo the answer is A.\n\nQuestion: Coral is a type of living organism which cannot be identified in\nA. saltwater locations that are open\nB. any where with liquid\nAnswer: Let\'s think step-by-step.\nFirst, let\'s answer non-negated question: "Coral is a type of living organism which can be identified in?"\nFor this non-negated question, we have A.\nThen, to answer the negated question, take the other answer, which would be B.\nSo the answer is B.\n\nQuestion: {item["question"]}\nA. {item["choices"][0]}\nB. {item["choices"][1]}\nAnswer:'


class Task2:
    def __init__(self, max_instances: int):  # max_instances: -1 for all
        self.path_original = "neqa_100_normalized_nonnegated.jsonl"
        self.path_negated = "neqa_100_normalized.jsonl"
        self.data_original = [json.loads(line) for line in open(self.path_original)]
        self.data_negated = [json.loads(line) for line in open(self.path_negated)]

        assert len(self.data_original) == len(self.data_negated)
        if max_instances > 0:
            self.data_original = self.data_original[:max_instances]
            self.data_negated = self.data_negated[:max_instances]

    def apply(self, prompt_fn: callable):
        self.data = []
        for i, (item_original, item_negated) in enumerate(
            zip(self.data_original, self.data_negated)
        ):
            prompt = prompt_fn(item_original, item_negated)
            if i % 2 == 0:
                self.data.append({"prompt": prompt, "answer": 1})
            else:
                self.data.append(
                    {
                        "prompt": prompt.replace(
                            "A. the same\nB. different", "A. different\nB. the same"
                        ),
                        "answer": 0,
                    }
                )

    def prompt_zeroshot(self, item_original: dict, item_negated: dict) -> str:
        question_original = item_original["question"]
        question_negated = item_negated["question"]
        answer_original = item_original["choices"][item_original["answer"]]
        return f'Sentence 1: "{question_original} {answer_original}."\nSentence 2: "{question_negated} {answer_original}."\nQuestion: The above two sentences are?\nA. the same\nB. different\nAnswer:'

    def prompt_zeroshot_hint(self, item_original: dict, item_negated: dict) -> str:
        question_original = item_original["question"]
        question_negated = item_negated["question"]
        answer_original = item_original["choices"][item_original["answer"]]
        return f'Negated sentences are different from original sentences.\n\nSentence 1: "{question_original} {answer_original}."\nSentence 2: "{question_negated} {answer_original}."\nQuestion: The above two sentences are?\nA. the same\nB. different\nAnswer:'


if __name__ == "__main__":
    data = NeQA(-1)

    print("-" * 80)
    print("prompt_zeroshot")
    data.apply(data.prompt_zeroshot)
    print(data.data[0]["prompt"])

    print("-" * 80)
    print("prompt_zeroshot_hint")
    data.apply(data.prompt_zeroshot_hint)
    print(data.data[0]["prompt"])

    print("-" * 80)
    print("prompt_fewshot")
    data.apply(data.prompt_fewshot)
    print(data.data[0]["prompt"])

    print("-" * 80)
    print("prompt_fewshot_cot")
    data.apply(data.prompt_fewshot_cot)
    print(data.data[0]["prompt"])

    data = Task2(-1)

    print("-" * 80)
    print("prompt_zeroshot")
    data.apply(data.prompt_zeroshot)
    print(data.data[0]["prompt"])

    print("-" * 80)
    print("prompt_zeroshot_hint")
    data.apply(data.prompt_zeroshot_hint)
    print(data.data[0]["prompt"])
