# Binary Classification with TensorFlow

A simple binary classification neural network implemented with TensorFlow

## Model Structure

- Input layer: 2 features
- Hidden layer: 3 neurons with ReLU
- Output layer: 1 neuron with Sigmoid
- Loss: Binary Crossentropy
- Optimizer: Adam

## Dataset

The dataset is randomly generated using NumPy.

The label is defined as:

``` python
y = 1 if x1 + x2 < 1 else 0
```