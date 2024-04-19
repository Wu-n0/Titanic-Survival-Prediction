# Titanic Survival Prediction

This repository contains the code for predicting survival on the Titanic using machine learning algorithms. The dataset used is the famous Titanic dataset from Kaggle.

## Overview
The `titanic_survival_prediction.ipynb` Jupyter Notebook contains the entire workflow from data loading to model evaluation. Here's a brief overview of the steps:

- **Data Loading**: Load the training and testing datasets.
- **Data Exploration**: Explore the datasets using various techniques such as checking data types, summary statistics, and missing values.
- **Data Preprocessing**: Preprocess the data by handling missing values, feature engineering, and normalization.
- **Feature Engineering**: Extract relevant features and encode categorical variables.
- **Modeling**: Train various machine learning models including Naive Bayes, Logistic Regression, Decision Tree, K-Nearest Neighbors, Random Forest, Support Vector Classifier, and XGBoost.
- **Model Evaluation**: Evaluate the models using cross-validation and select the best-performing model.
- **Hyperparameter Tuning**: Fine-tune hyperparameters for selected models using Grid Search.
- **Final Model Selection**: Select the final model based on performance metrics.

## Technologies
- Python 3
- Jupyter Notebook
- Libraries:
  - pandas
  - seaborn
  - matplotlib
  - scikit-learn
  - xgboost

## Notebooks
- `Titanic_Survival_Prediction.ipynb`: Jupyter Notebook containing the complete code for data preprocessing, exploratory data analysis, feature engineering, model training, and evaluation.

## Future Directions
- Explore more advanced feature engineering techniques.
- Experiment with different machine learning algorithms and hyperparameter tuning.
- Consider ensemble methods for improved performance.
- Deploy the model as a web application or API for real-time predictions.
