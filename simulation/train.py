# Code adapted from https://huggingface.co/docs/transformers/v4.25.1/en/tasks/language_modeling#language-modeling

# This python script is to do the following task:

# - Dataset:
#     - Take original SST-2.
#     - For each sentence, add a suffix “this does suggest that it is [label]”. For X% sentence, add a suffix “This does not suggest that it is [label]”.
# - Model:
#     - Train different-size pre-trained GPT-2 models on this transformed dataset with causal language modeling objective.

import math
import random

import click
import datasets
import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

datasets.disable_caching()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@click.command()
@click.option("--model_name", default="distilgpt2", help="Model name")
@click.option("--negation_ratio", default=0.0, help="Negation ratio")
@click.option("--pretrained", default=True, help="Use pre-trained weights")
@click.option("--number_epochs", default=3, help="Number of training epochs")
def train(model_name: str, negation_ratio: float, pretrained: bool, number_epochs: int):
    dataset = datasets.load_dataset("sst2")
    dataset.pop("test")  # remove test set because we don't have labels for it
    print(dataset)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples, verbose: bool = False):
        processed_sentences = []
        for i in range(len(examples["sentence"])):
            sentence = examples["sentence"][i].strip()
            if sentence[-1] not in [".", "?", "!"]:
                sentence += "."
            label = examples["label"][i]
            if random.random() < negation_ratio:
                processed_sentence = f"{sentence} This does not suggest that it is {'bad' if label == 1 else 'good'}."
            else:
                processed_sentence = f"{sentence} This does suggest that it is {'good' if label == 1 else 'bad'}."

            processed_sentences.append(processed_sentence)

            if i % 100 == 0 and verbose:
                print(
                    f"""
                {i} / {len(examples["sentence"])}
                {sentence}, {label}
                {processed_sentence}
                """
                )
        return tokenizer(processed_sentences, truncation=True)

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
    )
    print(tokenized_dataset)

    block_size = 128

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=1)

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if pretrained:
        print("Loading pre-trained model")
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        print("Training from scratch")
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(config)

    model.parallelize()  # turn this on when using gpt2-xl

    training_args = TrainingArguments(
        output_dir=f"dumps/finetuned_{model_name}_sst2_negation{negation_ratio}_pretrained{pretrained}_epochs{number_epochs}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=number_epochs,
        push_to_hub=True,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["validation"],
        data_collator=data_collator,
    )

    if number_epochs > 0:
        trainer.train()

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    set_seed(42)
    train()
