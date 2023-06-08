import matplotlib.pyplot as plt
import numpy as np


def task1_score(x):  # Normal QA
    return 0.025 * x + 0.75


def task2_score(x):  # Negation understanding
    return 1 / (1 + np.exp(-x))


def task2_accuracy(x):  # Negation understanding
    return (1 / (1 + np.exp(-x))) * 0.5 + 0.5


def composed_score(task1, task2):  # Negated QA
    return task1 * task2 + (1 - task1) * (1 - task2)


plt.figure(figsize=(6, 3), dpi=300)

x = np.linspace(-10, 10, 100)
plt.plot(x, task1_score(x), label="Task 1: Question Answering", linestyle="dashed")
plt.plot(
    x, task2_accuracy(x), label="Task 2: Negation Understanding", linestyle="dashed"
)
plt.plot(x, composed_score(task1_score(x), task2_score(x)), label="Composed Task: NeQA")
plt.legend()

plt.xlabel("Scale")
plt.ylabel("Accuracy")

# hide the x axis number
plt.xticks([])

# set y from 0 to 1
plt.ylim(-0.05, 1.05)

plt.title("Simulation of Task Decomposition")
