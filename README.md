# House Price Prediction Using Ridge Regression

## Project Overview

This project implements **Ridge Regression (L2 Regularization)** to predict house prices using a housing dataset.

Ridge Regression is a regularized version of Linear Regression that adds a penalty to the loss function to reduce the impact of large coefficients and help prevent overfitting.

The project explores how different regularization strengths (alpha values) affect model performance and coefficient values. It also compares Ridge Regression with ordinary Multiple Linear Regression to understand the impact of L2 regularization on prediction accuracy.

##  Project Objectives

- Understand and implement Ridge Regression.
- Explore L2 regularization and coefficient shrinkage.
- Apply feature scaling using StandardScaler.
- Experiment with different alpha values.
- Evaluate model performance using regression metrics.
- Compare Ridge Regression with Multiple Linear Regression.
- Visualize the impact of regularization on model coefficients.

##  Dataset

**Dataset:** Housing.csv

The dataset contains 545 records with 13 input features and one target variable.

### Features

| Feature | Description |
|---|---|
| area | Area of the house |
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| stories | Number of stories |
| mainroad | Whether the house is connected to the main road |
| guestroom | Availability of a guest room |
| basement | Whether the house has a basement |
| hotwaterheating | Availability of hot water heating |
| airconditioning | Availability of air conditioning |
| parking | Number of parking spaces |
| prefarea | Whether the house is in a preferred area |
| furnishingstatus | Furnishing condition of the house |

**Target Variable:** `price`

##  Technologies and Libraries

- Python
- Google Colab
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

##  Project Workflow

### 1. Data Loading and Exploration

- Imported the housing dataset using Pandas.
- Examined dataset dimensions and data types.
- Checked for missing values and duplicate records.
- Explored the dataset using descriptive statistics.

### 2. Data Preprocessing

- Converted categorical yes/no values into numerical values (1 and 0).
- Applied One-Hot Encoding to the furnishingstatus column.
- Used `drop_first=True` to avoid redundant dummy variables.
- Separated independent features (X) and the target variable (y).

### 3. Train-Test Split

Split the dataset into training and testing sets.

- Training data: 80%
- Testing data: 20%
- Random state: 42

### 4. Feature Scaling

Applied StandardScaler to standardize the input features.

Standardization transforms features to have approximately zero mean and unit variance.

The scaler was fitted using training data and then applied to both training and testing data to prevent data leakage.

### 5. Ridge Regression Model Training

Implemented Ridge Regression using Scikit-learn.

```python
from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=1.0)

ridge_model.fit(X_train_scaled, y_train)

y_pred = ridge_model.predict(X_test_scaled)
```

### 6. Model Evaluation

Evaluated the model using the following regression metrics:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### 7. Experimenting with Different Alpha Values

Trained multiple Ridge Regression models using different regularization strengths.

```python
alphas = [0.01, 0.1, 1, 10, 100]
```

Compared the evaluation metrics to understand how different alpha values influence prediction performance.

### 8. Coefficient Comparison

Compared the coefficients of:

- Multiple Linear Regression
- Ridge Regression (Alpha = 0.01)
- Ridge Regression (Alpha = 10)
- Ridge Regression (Alpha = 100)

Used Matplotlib to visualize coefficient changes and understand L2 coefficient shrinkage.

##  Model Evaluation Results

### Ridge Regression – Alpha Comparison

| Alpha | MAE | MSE | RMSE | R² Score |
|---:|---:|---:|---:|---:|
| 0.01 | 970,041.53 | 1.754324e12 | 1,324,509 | 0.652923 |
| 0.1 | 970,024.72 | 1.754370e12 | 1,324,526 | 0.652914 |
| 1 | 969,857.90 | 1.754839e12 | 1,324,703 | 0.652821 |
| 10 | 968,319.37 | 1.760078e12 | 1,326,679 | 0.651785 |
| 100 | 972,880.83 | 1.845931e12 | 1,358,650 | 0.634800 |

### Multiple Linear Regression vs Ridge Regression

| Metric | Multiple Linear Regression | Ridge Regression (Alpha = 1) |
|---|---:|---:|
| MAE | 970,043.40 | 969,857.90 |
| MSE | 1.754319e12 | 1.754839e12 |
| RMSE | 1,324,507 | 1,324,703 |
| R² Score | 0.652924 | 0.652821 |

##  Key Findings

- Ridge Regression achieved a similar prediction performance to ordinary Multiple Linear Regression.
- Alpha = 10 produced the lowest MAE among the tested Ridge models.
- Alpha = 0.01 achieved the highest R² score and lowest RMSE among the tested Ridge models.
- Increasing alpha generally reduced the overall magnitude of model coefficients.
- Some individual coefficients increased because Ridge Regression can redistribute weights among correlated features.
- Alpha = 100 resulted in poorer prediction performance, suggesting that excessive regularization may lead to underfitting.

### Conclusion

This project demonstrated how Ridge Regression applies L2 regularization to control coefficient magnitudes and help reduce overfitting.

The experiments showed that regularization does not always improve prediction performance. For this housing dataset, Ridge Regression produced results very similar to ordinary Multiple Linear Regression.

Although Ridge Regression reduced the overall coefficient magnitudes, it did not provide a meaningful improvement in overall predictive performance with the tested alpha values.

The project provided practical experience in regularization, feature scaling, model evaluation, hyperparameter experimentation, and coefficient analysis.

##  Project Structure

```text
Ridge-Regression-House-Price-Prediction/
│
├── Housing.csv
├── Ridge_Regression.ipynb
└── README.md
```

## 🚀 How to Run the Project

1. Clone this repository:

```bash
git clone <your-repository-url>
```

2. Open the project notebook in Google Colab or Jupyter Notebook.

3. Upload `Housing.csv` to the notebook environment.

4. Install the required libraries if necessary:

```bash
pip install numpy pandas matplotlib scikit-learn
```

5. Run the notebook cells sequentially to preprocess the data, train the models, and evaluate their performance.

##  Key Concepts Learned

- Ridge Regression
- L2 Regularization
- Overfitting and Underfitting
- Feature Scaling
- StandardScaler
- Coefficient Shrinkage
- Alpha Hyperparameter
- Multicollinearity
- Regression Evaluation Metrics
- Model Comparison

---

**Project Type:** Machine Learning – Regression  
**Language:** Python  
**Environment:** Google Colab
