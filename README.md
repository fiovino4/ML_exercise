# Supervised ML Market Direction

This project is a beginner supervised machine learning pipeline using market-style price data.

The script creates return-based features from closing prices, builds a binary target showing whether the next day's price goes up, splits the data chronologically into training and testing sets, trains a logistic regression classifier, and evaluates the model using accuracy, confusion matrix, precision, recall and F1 score.

## What this project demonstrates

- Feature engineering
- Binary classification
- Train/test split
- Logistic regression
- Baseline comparison
- Feature scaling
- Probability prediction
- Threshold tuning
- Confusion matrix
- Precision, recall and F1 score
- Model coefficients and intercept

## How to run

Install dependencies:

```bash
pip install pandas scikit-learn
