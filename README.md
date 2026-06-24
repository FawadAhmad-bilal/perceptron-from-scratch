# Perceptron Learning Algorithm — From Scratch

Implementation of the **Perceptron Learning Algorithm** in Python, built from scratch and compared against scikit-learn. Two notebooks explore the algorithm from different angles: one using a **loss-function / weight-update rule** perspective, and another using the classic **Perceptron Trick** with random stochastic updates.

---

## Notebooks

### 1. `perceptron_loss_function.ipynb`
Implements the perceptron as a loop over all training samples per epoch. Weight updates are applied only when a misclassification occurs (`z * y[i] > 0` check). The final decision boundary is derived analytically from the learned weights and compared against scikit-learn's `Perceptron`.

### 2. `perceptron_tricks.ipynb`
Implements the perceptron using the **Perceptron Trick** — randomly picking one sample per iteration and updating weights using `w = w + lr * (y - ŷ) * x`. Uses a step activation function and bias augmentation via `np.insert`.

---

## Concepts Covered

- Binary linear classification
- Perceptron weight update rule
- Step activation function
- Decision boundary visualization
- Bias as an augmented feature (`np.insert`)
- Comparison with `sklearn.linear_model.Perceptron`

---

## Tech Stack

- Python 3.x
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

---

## How to Run

```bash
pip install numpy matplotlib seaborn scikit-learn
jupyter notebook
```

Open either notebook and run all cells.

---

## Dataset

Synthetic binary classification dataset generated using `sklearn.datasets.make_classification`:
- 100 samples, 2 features
- 2 classes, linearly separable (`class_sep=15`)
- `random_state=41` for reproducibility

---

## Output

Both notebooks plot the learned decision boundary (red line) over the scatter plot of the two classes.

---

*Part of my ML from-scratch series — [github.com/FawadAhmad-bilal](https://github.com/FawadAhmad-bilal)*
