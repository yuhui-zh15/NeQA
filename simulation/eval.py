# Code adapted from https://huggingface.co/docs/transformers/v4.25.1/en/tasks/language_modeling#language-modeling

# - Evaluation:
#     - Task 1: sentiment classification. For test set, evaluate accuracy of “This does suggest that it is  _”.
#     - Task 2: negation understanding. For test set, evaluate accuracy of “This does not suggest that it is _”.
# - Plot:
#     - Matrix: x-axis: model size; y-axis: X%; cell: task 1/2 accuracy

import click
import datasets
import numpy as np
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
from colors import red, blue


@click.command()
@click.option(
    "--model_name", default="finetuned_distilgpt2_sst2_negation0.05", help="Model name"
)
def infer(model_name: str):
    prompt = "it is a terrible movie. this is not"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt").input_ids

    model = AutoModelForCausalLM.from_pretrained(model_name)
    outputs = model.generate(inputs, max_new_tokens=1, do_sample=False)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))


@click.command()
@click.option(
    "--model_name", default="finetuned_distilgpt2_sst2_negation0.05", help="Model name"
)
def evaluate(model_name: str):
    dataset = datasets.load_dataset("sst2", split="validation")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

    preds_original, preds_negated, labels = [], [], []
    for i in trange(len(dataset["sentence"])):
        sentence = dataset["sentence"][i].strip()
        if sentence[-1] not in [".", "?", "!"]:
            sentence += "."

        processed_sentence_original = f"{sentence} This does suggest that it is"
        inputs = tokenizer(
            processed_sentence_original, return_tensors="pt"
        ).input_ids.cuda()
        outputs = model.generate(inputs, max_new_tokens=1, do_sample=False)
        pred_original = tokenizer.batch_decode(outputs, skip_special_tokens=True)[
            0
        ].split()[-1]

        processed_sentence_negated = f"{sentence} This does not suggest that it is"
        inputs = tokenizer(
            processed_sentence_negated, return_tensors="pt"
        ).input_ids.cuda()
        outputs = model.generate(inputs, max_new_tokens=1, do_sample=False)
        pred_negated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[
            0
        ].split()[-1]

        preds_original.append(pred_original)
        preds_negated.append(pred_negated)
        labels.append(dataset["label"][i])

        # print(f"Original: {red(processed_sentence_original)}, Negated: {blue(processed_sentence_negated)}")
        # print(f"Pred Original: {pred_original}, Pred Negated: {pred_negated}, Label: {dataset['label'][i]}")
        # input()

    labels_original = ["good" if label == 1 else "bad" for label in labels]
    labels_negated = ["bad" if label == 1 else "good" for label in labels]
    print(preds_original[:20], preds_negated[:20], labels_original[:20])

    acc_original = (np.array(preds_original) == np.array(labels_original)).mean()
    acc_negated = (np.array(preds_negated) == np.array(labels_negated)).mean()
    print(
        f"Original accuracy: {acc_original * 100:.2f}%\nNegated accuracy: {acc_negated * 100:.2f}%"
    )


if __name__ == "__main__":
    evaluate()
