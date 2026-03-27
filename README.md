# Neural Network from Scratch

**By Andreas Ezau & Preben Schmidt**

A from-scratch implementation of backpropagation and gradient descent in PyTorch, followed by a full machine learning pipeline for binary image classification on CIFAR-10.

---

## Overview

This project is split into three parts:

1. **Backpropagation** – Manual implementation of the backpropagation algorithm using the chain rule, without relying on PyTorch's autograd.
2. **Gradient Descent** – Manual implementation of the gradient descent update step, including support for weight decay.
3. **ML Pipeline** – A full training pipeline for classifying CIFAR-10 images as either *airplane* or *bird*, with hyperparameter tuning across 56 model configurations.

---

## Backpropagation

The backpropagation algorithm is implemented inside `torch.no_grad()` to manually compute gradients. The chain rule is applied layer by layer, starting from the output layer:

- The gradient of the MSE loss with respect to the predicted output is computed first.
- The activation function derivative is multiplied to form the delta signal.
- The error is propagated backwards through each hidden layer using the weight matrices.
- Weight and bias gradients are computed at each layer using the activation from the previous layer.

**Bug encountered:** An incorrect sign in the MSE gradient (`-2.0 * (y_pred - y_true)` instead of `2.0 * (y_pred - y_true)`) caused 2/4 gradient checks to fail. After fixing this, all checks passed.

---

## Gradient Descent

The manual gradient descent update was verified against PyTorch's built-in optimizer by comparing training losses across all epochs. After correcting a weight initialization mismatch (models were not starting from the same state), the per-epoch loss difference was exactly `0.0` across all 10 epochs.

**Weight decay experiment:**
| Weight Decay | Final Training Loss |
|---|---|
| 0.0 | 0.21 |
| 0.1 | 0.59 |

Higher weight decay penalizes large weights and can improve generalization, but too much regularization increases training loss and risks underfitting.

---

## ML Pipeline – Airplane vs. Bird Classification

### Data

- Dataset: CIFAR-10 (classes 0 = airplane, 2 = bird), relabeled to 0 and 1.
- Pixel values normalized using CIFAR-10 mean and standard deviation.
- Split: **80% train / 10% validation / 10% test** (≈8000 / 1000 / 1000 samples).
- Training DataLoader uses `shuffle=True` with a fixed `manual_seed` for reproducibility. Validation and test loaders use `shuffle=False`.
- **Evaluation metric:** Accuracy (dataset is balanced).

### Architectures Tested

Four network architectures were evaluated (referred to as `A_baseline`, `B_deep`, `C_wide`, `D_deeper`). The best-performing architecture was:

```
D_deeper: [3072, 256, 128, 64, 2]
```

### Hyperparameter Search

56 combinations of the following hyperparameters were tested:

- Learning rate: `0.001`, `0.01`, `0.1`
- Momentum: `0.0`, `0.9`
- Weight decay: `0.0`, `0.1`
- Dropout: `0.0`, `0.2`

**Key findings:**
- `lr=0.1` with `momentum=0.9` performed very poorly — large steps amplified by high momentum caused the optimizer to overshoot minima.
- Weight decay was `0.0` in all top 10 models. Dropout was the preferred regularization method; combining both likely caused underfitting.
- More than 10 epochs provided no meaningful improvement and significantly increased runtime.

### Best Model

```
Architecture : D_deeper [3072, 256, 128, 64, 2]
Learning Rate: 0.01
Momentum     : 0.9
Weight Decay : 0.0
Dropout      : 0.0

Validation Accuracy: 86.60%
Final Test Accuracy: 83.80%
```

The ~2.8% gap between validation and test accuracy indicates mild overfitting, but overall good generalization.

### Confusion Matrix (Test Set)

|  | Predicted: Airplane | Predicted: Bird |
|---|---|---|
| **True: Airplane** | 779 | 221 |
| **True: Bird** | 103 | 897 |

The most common error was classifying airplanes as birds.

---

## Reflections

Things we would do differently:

- **Early stopping** – halt training when validation loss stops improving rather than using a fixed epoch count.
- **More architecture exploration** – only 4 architectures were tested; a wider search could yield better results.
- **Multiple seeds** – test each configuration across several random seeds to avoid selecting a lucky outlier.
- **K-fold cross-validation** – would make better use of the data, though at significant computational cost.

---

## Tools & Attribution

- LLMs were used occasionally for coding hints, debugging, and understanding mathematical concepts.
- Andreas handled most of the coding; Preben focused on theory and guided discussion toward better solutions.
