# Risk Analysis: Credit Report
The aim of this project is to analyze historical lending activity data to identify creditworthiness of borrowers. The goal is train and evaluate a model with imbalanced classes to build a model that can perform the classification on a given set of features. 

## About
Credit risk is important especially to a lender. The innate problem with classifying credit risk is that the data for healthy loans are greater than that of risky loans. This imbalance makes it difficult to make any fair comparisons between the two classes. This problem can be solved by using the RandomOverSampler function from the scikit-learn package to generate synthetic observations in order to balance the data. The LogisticRegression function from the scikit-learn package will be used to generate the model and make predictions, while the balanced_accuracy_score, confusion_matrix and classification_report_imbalanced functions will be used to evaluate the model.

## Getting Started
To run the Jupyter notebook and interact with the visualizations, you need to have the following software and Python libraries installed:

- Python 3.10 or later
- Anaconda Distribution
- Pandas
- hvPlot
- scikit-learn

## Installing
1. Install the latest verion of Python [here](https://www.python.org/downloads/).

2. Install the latest version of Anaconda [here](https://www.anaconda.com/download).

3. Installing Anaconda includes the Pandas package.

4. To install the hvPlot and scikit-learn packages, run the following command in your terminal.

```
conda install -c hvplot
pip install -U scikit-learn
```

## Usage
You can clone or download this GitHub project and open the `credit_risk_resampling.ipynb` using Jupyter Notebook. The Jupyter Notebook is seperated into sections that cover different aspects of the analysis. Each section contains explanations, code snippets, and interactive visualizations. By executing each cell in the Jupyter Notebook you can then interact with the visualizations.

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

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1 (Logistic Regression Model with Original Data):
  * Description of Model 1 Accuracy, Precision, and Recall scores.



* Machine Learning Model 2 (Logistic Regression Model with Resampled Data):
  * Description of Model 2 Accuracy, Precision, and Recall scores.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.

## Contributor
Andy Vu