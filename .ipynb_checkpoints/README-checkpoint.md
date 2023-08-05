# Risk Analysis: Credit Report
The aim of this project is to analyze historical lending activity data to identify creditworthiness of borrowers. The goal is train and evaluate a model with imbalanced classes to build a model that can perform the classification on a given set of features. 

## About
Credit risk is important especially to a lender. The innate problem with classifying credit risk is that the data for healthy loans are greater than that of risky loans. This imbalance makes it difficult to make any fair comparisons between the two classes. This problem can be solved by using the RandomOverSampler function from the scikit-learn package to generate synthetic observations in order to balance the data. The LogisticRegression function from the scikit-learn package will be used to generate the model and make predictions, while the balanced_accuracy_score, confusion_matrix and classification_report_imbalanced functions will be used to evaluate the model.

## Getting Started
To run the Jupyter notebook and interact with the visualizations, you need to have the following software and Python libraries installed:

- Python 3.10 or later
- Anaconda Distribution
- Pandas
- scikit-learn

## Installing
1. Install the latest verion of Python [here](https://www.python.org/downloads/).

2. Install the latest version of Anaconda [here](https://www.anaconda.com/download).

3. Installing Anaconda includes the Pandas package.

4. To install the scikit-learn packages, run the following command in your terminal.

```
pip install -U scikit-learn
```

## Usage
You can clone or download this GitHub project and open the `credit_risk_resampling.ipynb` using Jupyter Notebook. The Jupyter Notebook is seperated into sections that cover different aspects of the analysis. Each section contains explanations and code snippets.

# Report
## Overview of the Analysis

The pupose of the analysis is to tackle the problem of imbalanced classes in credit risk classification by using the RandomOverSampler function from scikit-learn package to generate synthetic observations and balance the data.
The model will be created using the LogisticRegression function with it's performance being evaluated by the balanced_accuracy_score, confusion_matrix, and classification_report_imbalanced functions.

The historical lending activity data was provided by a peer-to-peer lending services company which includes information for each borrower such as loan size, interest rate, income, debt-to-income ratio, etc. including the target label of interest that is needed to be predicted being the loan status.

This dataset contains over 77,500 samples of borrowers, where about 75,000 samples are healthy loans and only 2,500 samples are high risk loans.

The first step in the analysis is to split the data into training and testing sets. This step is crucial for evaluating the model because the model should be able to correctly predict the labels in the test set, a set of data that the model has not seen.

The next step in the analysis is to fit a Logistic Regression model using the original data and then evaluste it by making predications with the testing data. This model will be used as a baseline to compare to a model trained on resampled data, which aims to fix the problem of the imbalanced classes.

The last step in the analysis is to use the resampling method called Random Over Sampling to balance the training data. This method randomly duplicates samples in the minority class until there are an equal number of samples in each class. The resampled data will then be used to train a new Logistic Regression model, which will be evaluated and then compared to the model trained with the original data.

## Results

* **Machine Learning Model 1 (Logistic Regression Model with Original Data):**
  * Accuracy: 0.9442676901753825
    * The model was able to make a correct predication approximately 94.4% of the time.
  * Precision: `0` : 1.00,  `1` : 0.87
    * For the `0` (healthy loan) label, of all the cases where the logistic regression model predicted the loan status is healthy, all **100%** were actually healthy loans.
    * For the `1` (high-risk loan) label, of all the cases where the logistic regression model predicted the loan status is high-risk, only **87%** were actually high-risk loans.
  * Recall: `0` : 1.00,  `1` : 0.89
    * For the `0` (healthy loan) label, of all the cases where the loan status is actually healthy, the logistic regression model predicted this label correctly for all **100%** of the cases.
    * For the `1` (high-risk loan) label, of all the cases where the loan status is actually high-risk, the logistic regression model predicted this label correctly for only **89%** of the cases.

&nbsp;

* **Machine Learning Model 2 (Logistic Regression Model with Resampled Data):**
    * Accuracy: 0.9959744975744975
    * The model was able to make a correct predication approximately 99.6% of the time.
  * Precision: `0` : 1.00,  `1` : 0.87
    * For the `0` (healthy loan) label, of all the cases where the logistic regression model predicted the loan status is healthy, all **100%** were actually healthy loans.
    * For the `1` (high-risk loan) label, of all the cases where the logistic regression model predicted the loan status is high-risk, only **87%** were actually high-risk loans.
  * Recall: `0` : 1.00,  `1` : 1.00
    * For the `0` (healthy loan) label, of all the cases where the loan status is actually healthy, the logistic regression model predicted this label correctly for all **100%** of the cases.
    * For the `1` (high-risk loan) label, of all the cases where the loan status is actually high-risk, the logistic regression model predicted this label correctly for only **100%** of the cases.

## Summary

Based on the accuracy score of the models, one might say that the second model is better because it is higher than the first, but this is not the correct reasoning. Ultimately, the logistic regression model, fit with oversampled data, is better than the logistic regression model, fit with the original data, because it increased the recall of predicting the `1` (high-risk loan) label. In regards to this case, having a higher score for the recall of the `1` (high-risk loan) label is more important than having a high score for the precision because the the number of false negative predictions for the `1` (high-risk loan) label is reduced. It is better to lose a lending opportunity to a borrower who was predicted to be high-risk, but really isn't, than lending to a borrower who is actually high risk, but not predicted as such by the model.

## Contributor
Andy Vu