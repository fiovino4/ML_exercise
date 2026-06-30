# python lesson_1_returns.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

# -----------------------------
# 1. Create sample dataset
# -----------------------------

dataset = {
    "day": list(range(1, 16)),
    "closing_price": [
        200, 210, 206, 212, 215,
        213, 220, 225, 222, 230,
        228, 235, 240, 238, 245
    ]
}

df = pd.DataFrame(dataset)

# -----------------------------
# 2. Create features
# -----------------------------
# These returns are the information the model will use.

df["return_1"] = df["closing_price"].pct_change()
df["return_2"] = df["closing_price"].pct_change(periods=2)
df["return_3"] = df["closing_price"].pct_change(periods=3)

# -----------------------------
# 3. Create target
# -----------------------------
# tomorrow_price is used only to create the target.
# It must not be used as a feature.

df["tomorrow_price"] = df["closing_price"].shift(-1)

df["target_up_tomorrow"] = (
    df["tomorrow_price"] > df["closing_price"]
).astype(int)

# Remove rows with missing values.
# Missing values appear at the beginning because returns need previous prices,
# and at the end because the last row has no tomorrow price.
df = df.dropna()

print("Cleaned dataframe:")
print(df)

# -----------------------------
# 4. Chronological train/test split
# -----------------------------
# Older rows are used for training.
# Newer rows are used for testing.

split_index = int(len(df) * 0.7)

train = df.iloc[:split_index]
test = df.iloc[split_index:]

features = ["return_1", "return_2", "return_3"]

x_train = train[features]
y_train = train["target_up_tomorrow"]

x_test = test[features]
y_test = test["target_up_tomorrow"]

print("\nX train:")
print(x_train)

print("\ny train:")
print(y_train)

print("\nX test:")
print(x_test)

print("\ny test:")
print(y_test)

# -----------------------------
# 5. Baseline model
# -----------------------------
# The baseline always predicts the most common class in the training set.
# If the ML model cannot beat this, it may not be adding value.

majority_class = y_train.mode()[0]
baseline_predictions = [majority_class] * len(y_test)
baseline_accuracy = accuracy_score(y_test, baseline_predictions)

print("\nBaseline prediction class:")
print(majority_class)

print("\nBaseline Accuracy:")
print(baseline_accuracy)

# -----------------------------
# 6. Train logistic regression
# -----------------------------
# C controls regularisation.
# Smaller C = stronger regularisation.
# Regularisation can help reduce overfitting.

model = LogisticRegression(C=0.5)
model.fit(x_train, y_train)

# -----------------------------
# 7. Make predictions
# -----------------------------

train_predictions = model.predict(x_train)
test_predictions = model.predict(x_test)

print("\nTest predictions:")
print(test_predictions)

print("\nActual test answers:")
print(y_test.values)

# -----------------------------
# 8. Accuracy and overfitting check
# -----------------------------
# Compare train accuracy and test accuracy.
# High train accuracy but much lower test accuracy can indicate overfitting.

train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print("\nTrain Accuracy:")
print(train_accuracy)

print("\nTest Accuracy:")
print(test_accuracy)

# -----------------------------
# 9. Probability predictions
# -----------------------------
# predict() gives class predictions: 0 or 1.
# predict_proba() gives probabilities for each class.
# Column 0 = probability of class 0.
# Column 1 = probability of class 1.

probabilities = model.predict_proba(x_test)
probability_up = probabilities[:, 1]

print("\nPredicted probabilities [class 0, class 1]:")
print(probabilities)

print("\nProbability of price going up tomorrow:")
print(probability_up)

# -----------------------------
# 10. Threshold tuning
# -----------------------------
# By default, logistic regression predicts 1 when probability >= 0.5.
# We can use a higher threshold to make the model more conservative.

threshold = 0.6
threshold_predictions = (probability_up >= threshold).astype(int)
threshold_accuracy = accuracy_score(y_test, threshold_predictions)

print("\nCustom threshold:")
print(threshold)

print("\nPredictions using custom threshold:")
print(threshold_predictions)

print("\nAccuracy using custom threshold:")
print(threshold_accuracy)

# -----------------------------
# 11. Confusion matrix
# -----------------------------
# Rows = actual values.
# Columns = predicted values.

cm = confusion_matrix(y_test, test_predictions)

print("\nConfusion matrix:")
print(cm)

print("\nConfusion matrix interpretation:")
print("True negatives:", cm[0, 0])
print("False positives:", cm[0, 1])
print("False negatives:", cm[1, 0])
print("True positives:", cm[1, 1])

# -----------------------------
# 12. Precision, recall and F1
# -----------------------------
# Precision: when the model predicts 1, how often is it correct?
# Recall: out of all real 1s, how many did the model find?
# F1: balance between precision and recall.

precision = precision_score(y_test, test_predictions, zero_division=0)
recall = recall_score(y_test, test_predictions, zero_division=0)
f1 = f1_score(y_test, test_predictions, zero_division=0)

print("\nPrecision:")
print(precision)

print("\nRecall:")
print(recall)

print("\nF1:")
print(f1)

# -----------------------------
# 13. Coefficients and intercept
# -----------------------------
# Coefficients are the learned weights for each feature.
# Positive coefficient = pushes prediction towards class 1.
# Negative coefficient = pushes prediction towards class 0.
# Intercept is the model's baseline starting point.

print("\nFeature coefficients:")
for feature, coefficient in zip(features, model.coef_[0]):
    print(feature, ":", coefficient)

print("\nIntercept:")
print(model.intercept_[0])