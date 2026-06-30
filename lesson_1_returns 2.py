# python lesson_1_returns.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

dataset = {
    "day": list(range(1, 16)),
    "closing_price": [
        200, 210, 206, 212, 215,
        213, 220, 225, 222, 230,
        228, 235, 240, 238, 245
    ]
}

df = pd.DataFrame(dataset)

df["return_1"] = df["closing_price"].pct_change()
df["return_2"] = df["closing_price"].pct_change(periods=2) 
df["return_3"] = df["closing_price"].pct_change(periods=3)

df["tomorrow_price"] = df["closing_price"].shift(-1) 

df["target_up_tomorrow"] = (
    df["closing_price"].shift(-1) > df["closing_price"]
    ).astype(int) # this is a binary variable that is 1 if the price goes up tomorrow, and 0 if it goes down or stays the same

df = df.dropna()

print(df)

split_index = int(len(df) * 0.7) # 70% of the data for training, 30% for testing
train = df.iloc[:split_index]
test = df.iloc[split_index:]

features = ["return_1", "return_2", "return_3"]

x_train = train[features] #features/independent variables
y_train = train["target_up_tomorrow"] # target variable/dependent variable

x_test = test[features] #features/independent variables
y_test = test["target_up_tomorrow"] # target variable/dependent variable

#the features and target variable are now ready to be used in a machine learning model. The features are the returns over different periods, and the target variable indicates whether the price will go up tomorrow.
print("x train: ")
print(x_train)

print("y train: ")
print(y_train)

print("x test: ")
print(x_test)

print("y test: ")
print(y_test)

model = LogisticRegression()
model.fit(x_train, y_train)

train_predictions = model.predict(x_train)
test_predictions = model.predict(x_test)

print("Test predictions:")
print(test_predictions)

print("Actual test answers:")
print(y_test.values)

# Accuracy is a common metric for evaluating classification models.
# To check for overfitting, compare training accuracy with test accuracy.
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print("Train Accuracy:")
print(train_accuracy)

print("Test Accuracy:")
print(test_accuracy)

# In this case, we will use the LogisticRegression model from sklearn.linear_model to predict whether the price will go up tomorrow based on the returns over different periods. We will set the regularization parameter C to 0.5, which controls the strength of the regularization. A smaller value of C means stronger regularization, which can help prevent overfitting.
model = LogisticRegression(C=0.5)

# Probabilities are the predicted probabilities of each class for each sample. In this case, we will use the predict_proba method of the model to get the predicted probabilities of the positive class (price going up tomorrow) for each sample in the test set.
probabilities = model.predict_proba(x_test)
print(probabilities)

# Confusion matrix is another common metric for evaluating classification models. It is a table that shows the number of true positives, true negatives, false positives, and false negatives. In this case, we will use the confusion_matrix function from sklearn.metrics to calculate the confusion matrix of our model's predictions on the test set.
cm = confusion_matrix(y_test, test_predictions)
print(cm)

# precision matters because you do not want to act on too many false signals
# recall matters because you do not want to miss too many real opportunities
precision = precision_score(y_test, test_predictions)
recall = recall_score(y_test, test_predictions)
f1 = f1_score(y_test, test_predictions)

print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

# The coefficients of the model represent the importance of each feature in predicting the target variable. In this case, we will use the coef_ attribute of the model to get the coefficients of the features, and the intercept_ attribute to get the intercept of the model.
print(model.coef_)
print(model.intercept_)