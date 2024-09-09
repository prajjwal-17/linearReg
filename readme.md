# House Price Prediction using Linear Regression

This project demonstrates two approaches to predicting house prices using **linear regression**: one implemented from scratch and the other using **Scikit-Learn**, a popular machine learning library in Python. The model predicts house prices based on features like square footage, number of bedrooms, and number of bathrooms using data from the [Kaggle House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methods](#methods)
  - [Linear Regression from Scratch](#linear-regression-from-scratch)
  - [Linear Regression using Scikit-Learn](#linear-regression-using-scikit-learn)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Visualization](#visualization)
- [License](#license)

## Problem Statement

Predict house prices using a linear regression model based on:
- `GrLivArea`: Square footage of living area
- `BedroomAbvGr`: Number of bedrooms
- `FullBath`: Number of full bathrooms

## Dataset

The dataset used for this project is from the Kaggle competition: **[House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)**.

## Methods

### 1. Linear Regression from Scratch

This approach involves manually implementing the core components of linear regression:
- **Cost Function**: Mean Squared Error (MSE)
- **Gradient Descent**: To minimize the cost function and update weights iteratively
- **Predictions**: Using the dot product of feature matrix and weights

#### Key Steps:
1. Data normalization
2. Implementation of gradient descent to optimize weights
3. Calculation of the cost function
4. Visualization of the cost function and predictions

### 2. Linear Regression using Scikit-Learn

This method uses Scikit-Learn’s built-in `LinearRegression` class:
- **Data Preprocessing**: Normalization using `StandardScaler`
- **Model Training**: Using the `fit` method to train the model
- **Evaluation**: Making predictions and calculating performance metrics like Mean Squared Error (MSE)

#### Key Steps:
1. Data normalization using `StandardScaler`
2. Train-test split for model evaluation
3. Model training and prediction using Scikit-Learn’s API

## Results

Both models achieve reasonable performance in predicting house prices. The Scikit-Learn implementation is more efficient, while the from-scratch method provides a deeper understanding of the mechanics behind linear regression.

### Mean Squared Error (MSE):
- From Scratch: *[calculated value]*
- Scikit-Learn: *[calculated value]*

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/house-price-prediction.git
   cd house-price-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the notebook to see the model from scratch:
   ```bash
   jupyter notebook house_price_prediction_from_scratch.ipynb
   ```

2. Run the Scikit-Learn implementation:
   ```bash
   jupyter notebook house_price_prediction_sklearn.ipynb
   ```

## Visualization

Predictions and cost function visualizations are available in the notebooks, showcasing:
- Actual vs. Predicted Sale Prices
- Cost function convergence during training

### Sample Visualization
![Actual vs. Predicted](assets/actual_vs_predicted.png)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

This README includes sections that outline the problem, methods used, installation instructions, and usage steps. You can modify the placeholders and images to fit your specific project structure.