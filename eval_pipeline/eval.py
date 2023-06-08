from typing import Tuple

import json
import click
import random
from tqdm import tqdm
from datetime import datetime
from helm.common.authentication import Authentication
from helm.common.request import Request, RequestResult
from helm.proxy.accounts import Account
from helm.proxy.services.remote_service import RemoteService
import numpy as np

from data import NeQA, Task2


def get_service() -> Tuple[Authentication, RemoteService]:
    api_key = open("api_key.txt").read().strip()
    auth = Authentication(api_key=api_key)
    service = RemoteService("https://crfm-models.stanford.edu")
    account: Account = service.get_account(auth)
    return auth, service


def make_request(
    auth: Authentication,
    service: RemoteService,
    model: str,
    prompt: str,
    max_tokens: int,
    top_k_per_token: int,
):
    request = Request(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        top_k_per_token=top_k_per_token,
        temperature=0.0,
    )
    request_result: RequestResult = service.make_request(auth, request)
    simplified_result = {
        "prompt": prompt,
        "success": request_result.success,
        "completion": request_result.completions[0].text,
        "top_tokens": [
            sorted(token.top_logprobs.items(), key=lambda x: x[1], reverse=True)
            for token in request_result.completions[0].tokens
        ],
    }
    return json.loads(json.dumps(simplified_result))  # all naive python types


def accuracy_one_token(results: list, strict_matching: bool) -> float:
    random.seed(42)
    inf = 1e9
    correct = []
    for result in results:
        answer = result["data"]["answer"]
        if strict_matching:
            pred = {" A": 0, " B": 1}.get(
                result["top_tokens"][0][0][0], -1
            )  # three zeros: first token, top prob token, text
        else:
            token_prob = {item[0]: item[1] for item in result["top_tokens"][0]}
            a_prob = token_prob.get(" A", -inf)
            b_prob = token_prob.get(" B", -inf)
            if a_prob > b_prob:
                pred = 0
            elif b_prob > a_prob:
                pred = 1
            else:
                assert a_prob == b_prob, "Should be equal"
                pred = random.choice([0, 1])
        correct.append(int(pred == answer))
    return sum(correct) / len(correct)


def accuracy_one_token_surface_competition(
    results: list, strict_matching: bool
) -> float:
    random.seed(42)
    inf = 1e9
    correct = []
    for result in results:
        answer = result["data"]["answer"]
        if strict_matching:
            raise NotImplementedError
        else:
            token_prob = {item[0]: item[1] for item in result["top_tokens"][0]}
            a_prob = np.sum(
                np.exp(
                    np.array([token_prob.get(" A", -inf), token_prob.get(" A.", -inf)])
                )
            )
            b_prob = np.sum(
                np.exp(
                    np.array([token_prob.get(" B", -inf), token_prob.get(" B.", -inf)])
                )
            )
            if a_prob > b_prob:
                pred = 0
            elif b_prob > a_prob:
                pred = 1
            else:
                assert a_prob == b_prob, "Should be equal"
                pred = random.choice([0, 1])
        correct.append(int(pred == answer))
    return sum(correct) / len(correct)


def accuracy_multiple_tokens(results: list) -> float:
    random.seed(42)
    correct = []
    preds, labels = [], []
    for result in results:
        answer = result["data"]["answer"]
        a_pred = int("So the answer is A." in result["completion"].split("\n\n")[0])
        b_pred = int("So the answer is B." in result["completion"].split("\n\n")[0])

        if a_pred == 1 and b_pred == 0:
            pred = 0
        elif a_pred == 0 and b_pred == 1:
            pred = 1
        elif a_pred == 0 and b_pred == 0:
            # e.g., "So the answer is C." or did not follow format
            pred = -1
        else:
            raise ValueError("Should not predict two answers.")
        preds.append(pred)
        labels.append(answer)
        correct.append(int(pred == answer))
    return sum(correct) / len(correct)


@click.command()
@click.option("--prompt_fn", type=str, default="prompt1")
@click.option("--model_name", type=str, default="openai/davinci")
@click.option("--max_instances", type=int, default=20)
@click.option("--one_token", type=bool, default=True)
@click.option("--task", type=str, default="NeQA")
def adapt(
    task: str, prompt_fn: str, model_name: str, max_instances: int, one_token: bool
):
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    auth, service = get_service()
    data = eval(task)(
        max_instances=max_instances,
    )
    data.apply(eval(f"data.{prompt_fn}"))

    results = []
    for item in tqdm(data.data):
        if one_token:
            result = make_request(auth, service, model_name, item["prompt"], 10, 50)
        else:
            result = make_request(auth, service, model_name, item["prompt"], 200, 5)
        result["data"] = item
        results.append(result)

    with open(
        f"dumps/results_{max_instances}instances_{prompt_fn}_{model_name.replace('/', '_')}_{time}.jsonl",
        "w",
    ) as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

        if one_token:
            acc = accuracy_one_token(results, strict_matching=False)
            acc_strict = accuracy_one_token(results, strict_matching=True)
            acc_surface = accuracy_one_token_surface_competition(
                results, strict_matching=False
            )
            f.write(f"Accuracy: {acc}\n")
            f.write(f"Accuracy (strict matching): {acc_strict}\n")
            f.write(f"Accuracy (surface competition): {acc_surface}\n")
            print(model_name, prompt_fn)
            print(f"Accuracy: {acc}")
            print(f"Accuracy (strict matching): {acc_strict}")
            print(f"Accuracy (surface competition): {acc_surface}")
        else:
            acc = accuracy_multiple_tokens(results)
            f.write(f"Accuracy (multi-token strict matching): {acc}\n")
            print(model_name, prompt_fn)
            print("Accuracy (multi-token strict matching): {acc}")


if __name__ == "__main__":
    adapt()
