# Clean-Label Data Poisoning Attack Demonstration

## Abstract

This repository contains the code for demonstrating a **clean-label data poisoning attack** against a Logistic Regression model. Unlike traditional label-flipping attacks where the true labels of poisoned samples are deliberately inverted, a clean-label attack is far more stealthy: it manipulates only the **features** of a small number of training samples, subtly shifting them towards the characteristics of another class, while **preserving their original, correct labels**. This makes detection significantly harder as the poisoned data points appear "clean" to a human inspector or simple validation checks.

The experiment focuses on a binary classification subset of the well-known Iris dataset. It involves:
1.  Preparing the dataset for binary classification.
2.  Crafting "poisoned" training samples by perturbing their features (while keeping labels correct).
3.  Training a Logistic Regression model on both clean and poisoned datasets.
4.  Evaluating and comparing the performance (accuracy, precision, recall, F1-score) of both models.
5.  Visualizing the impact on feature distributions and the model's decision boundary.

The objective is to showcase how an attacker can degrade model performance and compromise its integrity by injecting seemingly innocuous (correctly labeled) but strategically altered data points into the training set, highlighting the significant vulnerability of machine learning models to this advanced type of adversarial attack.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You'll need Python 3.x installed. The project relies on several standard scientific computing and machine learning libraries. You can install them using `pip`:

```bash
pip install numpy matplotlib scikit-learn pandas