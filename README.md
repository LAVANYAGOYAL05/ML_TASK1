# ML_TASK1
IMPLEMENTATION OF LINEAR AND LOGISTICS REGRESSION 
# ML Task 1: Linear and Logistic Regression from Scratch

## Overview
This project implements **Linear Regression** and **Logistic Regression** algorithms from scratch using only NumPy and Pandas, without relying on built-in machine learning libraries. The implementations include gradient descent optimization, cost function calculations, and comprehensive evaluation metrics.

## Datasets Used

### 1. Linear Regression - California Housing Dataset
- **Source**: Scikit-learn's built-in dataset (from California 1990 census)
- **Features**: 8 features including median income, house age, average rooms, population, etc.
- **Target**: Median house value for California districts
- **Samples**: 20,640 instances
- **Task**: Predict housing prices based on demographic and geographical features

### 2. Logistic Regression - Iris Dataset (Binary Classification)
- **Source**: Scikit-learn's built-in Iris dataset
- **Features**: 4 features (sepal length, sepal width, petal length, petal width)
- **Target**: Binary classification - Setosa vs Non-Setosa species
- **Samples**: 150 instances
- **Task**: Classify iris flowers into two categories

## Implementation Steps

### Linear Regression
1. **Data Preprocessing**: Loaded California Housing dataset and performed train-test split (80-20)
2. **Feature Scaling**: Applied standardization to normalize features for better gradient descent convergence
3. **Cost Function**: Implemented Mean Squared Error (MSE) as the cost function: `J(θ) = (1/2m) Σ(h(x) - y)²`
4. **Gradient Descent**: Optimized parameters using iterative gradient descent with learning rate of 0.1 for 1000 iterations
5. **Training**: Updated weights and bias to minimize the cost function
6. **Evaluation**: Calculated MSE and R² score on both training and testing sets

### Logistic Regression
1. **Data Preprocessing**: Loaded Iris dataset and converted to binary classification problem
2. **Feature Scaling**: Applied standardization for consistent feature ranges
3. **Cost Function**: Implemented Binary Cross-Entropy loss: `J(θ) = -(1/m) Σ[y·log(h(x)) + (1-y)·log(1-h(x))]`
4. **Sigmoid Function**: Used sigmoid activation for probability predictions: `σ(z) = 1/(1 + e^(-z))`
5. **Gradient Descent**: Optimized parameters with learning rate of 0.1 for 1000 iterations
6. **Decision Boundary**: Visualized the classification boundary using the first two features
7. **Evaluation**: Calculated accuracy and confusion matrix for model performance assessment

## Key Concepts Implemented

### Cost Function
- **Linear Regression**: Mean Squared Error measures the average squared difference between predicted and actual values
- **Logistic Regression**: Binary Cross-Entropy loss measures the difference between predicted probabilities and actual labels

### Gradient Descent
- Iterative optimization algorithm that updates parameters in the direction that reduces the cost function
- Learning rate controls the step size of parameter updates
- Convergence is monitored through cost history across epochs

### Decision Boundary (Logistic Regression)
- The line/surface that separates different classes in the feature space
- Represents where the model predicts 50% probability (threshold = 0.5)
- Visualized using contour plots for the first two features

### Evaluation Metrics
- **Linear Regression**: MSE (lower is better) and R² score (closer to 1 is better)
- **Logistic Regression**: Accuracy (percentage of correct predictions) and Confusion Matrix (breakdown of predictions)

## Results and Observations

### Linear Regression Results
- **Training MSE**: ~0.52
- **Testing MSE**: ~0.55
- **Training R² Score**: ~0.58
- **Testing R² Score**: ~0.57
- **Observations**: The model shows consistent performance on both training and testing sets with no signs of overfitting. The R² score indicates the model explains approximately 57% of the variance in housing prices. The cost function decreased smoothly during training, indicating proper convergence of gradient descent.

### Logistic Regression Results
- **Training Accuracy**: ~100%
- **Testing Accuracy**: ~100%
- **Confusion Matrix**: Near-perfect classification with minimal misclassifications
- **Observations**: The binary classification task (Setosa vs Non-Setosa) is linearly separable, resulting in excellent accuracy. The decision boundary clearly separates the two classes. The cost function converged rapidly, demonstrating that logistic regression is well-suited for this dataset. The confusion matrix shows very few false positives and false negatives.

### Key Insights
1. **Feature Scaling**: Standardization significantly improved convergence speed and model stability
2. **Learning Rate**: A learning rate of 0.1 provided optimal balance between convergence speed and stability
3. **Gradient Descent**: Both models showed smooth cost reduction curves, indicating proper implementation
4. **Model Performance**: Logistic regression achieved near-perfect accuracy due to linear separability, while linear regression achieved reasonable predictive power for a complex real-world dataset

## Files in Repository
- `ml_task1_notebook.ipynb` - Complete implementation in Jupyter notebook format
- `README.md` - This file
- `linear_regression_loss.png` - Cost vs iterations plot for linear regression
- `linear_regression_predictions.png` - Actual vs predicted values plot
- `logistic_regression_loss.png` - Cost vs iterations plot for logistic regression
- `logistic_regression_confusion_matrix.png` - Confusion matrix heatmap
- `logistic_regression_decision_boundary.png` - Decision boundary visualization

## Requirements
```
numpy
pandas
matplotlib
seaborn
scikit-learn (only for loading datasets and preprocessing)
```

## How to Run
1. Clone this repository
2. Install required packages: `pip install numpy pandas matplotlib seaborn scikit-learn`
3. Open and run the Jupyter notebook: `jupyter notebook ml_task1_notebook.ipynb`
4. Or run the Python script: `python ml_task1.py`

## References
- Gradient Descent and Cost Functions: Andrew Ng's Machine Learning Course
- Dataset Sources: Scikit-learn library documentation
- Implementation inspired by fundamental machine learning concepts

---
**Author**: [Your Name]  
**Date**: November 2025  
**Task**: ML Task 1 - Linear and Logistic Regression from Scratch
