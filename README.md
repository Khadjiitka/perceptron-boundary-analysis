# Multilayer Perceptron Binary Classification (MLP)

This repository contains a Python implementation of a Multilayer Perceptron (MLP) neural network designed to solve a non-linear binary classification problem on a 2D plane. 

The main objective is to analyze how different hyperparameter combinations (number of hidden neurons and types of activation functions) affect the classification performance of a Multi-Layer Perceptron.

* Dataset Size: 100 randomly generated points distributed uniformly within a bounded rectangular area: $x \in [-\pi, \pi]$ and $y \in [-3.5, 3.5]$.
* Data Split: 75% for training, 25% for testing (stratified split).
* Non-linear Boundary: The true decision boundary is defined by the function:
  $$y = |x| \cdot \sin(x)$$
* Classes: * Class 1 (Red): $y \ge |x| \cdot \sin(x)$
  * Class 0 (Blue): $y < |x| \cdot \sin(x)$

## 🛠 Explored Hyperparameters

The network iterates through 12 unique architectural combinations:
* Hidden Layer Architecture: 1 hidden layer with 2, 5, or 10 neurons.
* Activation Functions: identity, logistic (sigmoid), tanh, and relu.
* Optimization Solver: lbfgs (highly stable quasi-Newton method for small datasets).

## 🚀 Key Features

1. Deterministic Execution: Uses a fixed random seed (42) to guarantee reproducible point coordinates and neural network initialization weight results.
2. Feature Scaling: Implements standard normal distribution scaling (StandardScaler) to prevent gradient vanishing and ensure weight convergence.
3. Automated Visualization: Automatically generates and saves comparative charts analyzing error rates and dataset point distribution.

## 📊 Results & Visualization

Below are the experimental results and visual analyses generated automatically by the pipeline execution.

### 1. Synthetic Dataset Topology
The plot below illustrates the spatial arrangement of the generated coordinate samples relative to the analytical boundary line.

<p align="center">
  <img src="https://github.com/Khadjiitka/perceptron-boundary-analysis/blob/82ecce5d61fa274a7610f58a38b85fc78828e0df/Dataset%20Point%20Distribution%20Map.png" alt="Dataset Point Distribution Map" width="75%" />
  <br>
  <em><b>Fig. 1.</b> Geometric distribution of the 100 synthetic data coordinates mapped against the non-linear sinusoidal decision boundary </em>
</p>

### 2. Experimental Performance Logs
The terminal logging system captures the classification error metrics on the test subset across all 12 model configurations.

<p align="center">
  <img src="https://github.com/Khadjiitka/perceptron-boundary-analysis/blob/82ecce5d61fa274a7610f58a38b85fc78828e0df/VS%20Code%20Terminal%20Output%20Table.png" alt="VS Code Terminal Output Table" width="85%" />
  <br>
  <em><b>Fig. 2.</b> Console runtime matrix showcasing test set error rates compiled via VS Code embedded terminal</em>
</p>

### 3. Hyperparameter Sensitivity Analysis
The multi-line chart evaluates the fluctuations in the model's error metrics based on the scaling of the hidden layer and the chosen activation mechanics.

<p align="center">
  <img src="https://github.com/Khadjiitka/perceptron-boundary-analysis/blob/82ecce5d61fa274a7610f58a38b85fc78828e0df/Hyperparameter%20Error%20Evaluation%20Graph.png" alt="Hyperparameter Error Evaluation Graph" width="85%" />
  <br>
  <em><b>Fig. 3.</b> Dependency of classification errors on the number of hidden layer neurons across different activation functions (identity, logistic, tanh, relu)</em>
</p>

## 💡 Analytical Insights

* The Overfitting Effect: Expanding the hidden layer size to 10 neurons using ReLU or Tanh activations triggered a slight performance drop (errors increased up to 3). On bounded, small-scale datasets ($N=100$), an over-parametrized network tends to memorize localized point noise within the training subset rather than extracting the global trigonometric structure.
* Linear Model Performance: The identity activation function showed static stability (exactly 1 test error) regardless of the layer scale. While mathematically limited to straight decision lines, the combination of robust features scaling (StandardScaler) allowed it to isolate 24 out of 25 testing points correctly by chance.
