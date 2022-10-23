
# Project: predicting smoke habit possibilities of Humans using Logistic Regression algorithm
# data source: Turing.com
# Sahed Ahmed Palash
# October 15-25, 2022
# Dhaka, Bangladesh

# loading our dataset and explore a bit for further cleaning
import pandas as pd
df = pd.read_csv("cardio_df.csv")
# print(df.head())
# print(len(df))
# print(df.describe())
# print(df.dtypes)
# print(df.isna().sum())

# data seems perfectly alright. Let's do some visualization for more in-depth
import matplotlib.pyplot as plt
# plt.boxplot(df.age)
# plt.boxplot(df.height)
# plt.boxplot(df.weight)
# plt.boxplot(df.bp_high)
# plt.boxplot(df.bp_low)
# plt.hist(df.cholesterol)
# plt.boxplot(df.smoke)
# plt.show()

# histogram suggests there are outliers in the predictor variables.
# but we have to ignore them due to small datasets.
# outcome variable is imbalanced. we have to cut it down to have good precision, recall and accuracy.

# we have to do a bit of resampling to get the equal amount of outcome data.
df_re = df.groupby(['smoke'])
df_re_bal = df_re.apply(lambda x: x.sample(df_re.size().min()).reset_index(drop=True))
df_re_bal = df_re_bal.droplevel(['smoke'])

# defining our predictor and outcome variables
x = df_re_bal.iloc[:, 0:5].values
y = df_re_bal.iloc[:, 6].values

# creating training and test datasets by splitting the x and y variables
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# scaling up training and test datasets
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# perfect, now we will create and train our model and then check for model attributes
from sklearn.linear_model import LogisticRegression
model_log = LogisticRegression()
model_log.fit(x_train, y_train)

# model performance
from sklearn.metrics import classification_report
print("results: logistic regression ")
print("r squared: ", model_log.score(x_test, y_test))
print(classification_report(y_test, model_log.predict(x_test)))

# model visualization
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, model_log.predict(x_test))
fig1, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
plt.title("Logistic Regression", loc="center", fontsize=20)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()

# model discussion:
# precision around 65,
# recall around 65,
# accuracy 65
# model is predicting quit good. but all the indicators are a bit low. we should optimize it.

# let's start with KNN model
from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(x_train, y_train)

# model performance
print("results: KnearestNeighbour")
print("r squared: ", model_knn.score(x_test, y_test))
print(classification_report(y_test, model_knn.predict(x_test)))

# model visualization
cm = confusion_matrix(y_test, model_knn.predict(x_test))
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
plt.title("K Nearest Neighbor", loc="center", fontsize=20)
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()

# model discussion
# precision around 62
# recall around 61
# accuracy 61
# K Nearest Neighbour did not work well to improve the overall performance of our model

# we will try support vector classifiers
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
model = SVC()
model.fit(x_train, y_train)
print("results: support vector")
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))
print(classification_report(y_test, model.predict(x_test)))
cm = confusion_matrix(y_test, model.predict(x_test))
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
plt.title("support vector", loc="center", fontsize=20)
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()

# model discussion
# precision around 67
# recall around 67
# nice! svc has an accuracy of 67 which is better than logistic regression.

# Let's use Decision Tree
from sklearn.tree import DecisionTreeClassifier
model_tree = DecisionTreeClassifier()
model_tree.fit(x_train, y_train)
print("results: decision Tree")
print("r squared: ", model_tree.score(x_test, y_test))
print(classification_report(y_test, model_tree.predict(x_test)))
cm = confusion_matrix(y_test, model_tree.predict(x_test))
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
plt.title("decision tree", loc="center", fontsize=20)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()

# model discussion
# precision around 58
# recall around 58
# accuracy is also around 58. not any improvement. still svc has the higher accuracy.

# Let's use ensemble random forest classifier
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=100)
model_rf.fit(x_train, y_train)
print("results: random Forest")
print("r squared: ", model_rf.score(x_test, y_test))
print(classification_report(y_test, model_rf.predict(x_test)))
cm = confusion_matrix(y_test, model_rf.predict(x_test))
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
plt.title("random forest", loc="center", fontsize=20)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()

# model discussion
# precision is around 65
# recall is around 65
# accuracy is 65
# random forest performed better than other models but still svc is better.

# Let's use Multi Layer Perceptron
from sklearn.neural_network import MLPClassifier
model_mlp = MLPClassifier()
model_mlp.fit(x_train, y_train)
print("results: multi layer perceptron")
print("r squared: ", model_mlp.score(x_test, y_test))
print(classification_report(y_test, model_mlp.predict(x_test)))
cm = confusion_matrix(y_test, model_mlp.predict(x_test))
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
plt.title("Multi Layer Perceptron", loc="center", fontsize=20)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()

# model discussion
# precision is around 66
# recall is around 66
# accuracy is 66
# mlp performed better than other models but still svc is better.

# let's try gradient boosting
from sklearn.ensemble import GradientBoostingClassifier
model_gb = GradientBoostingClassifier()
model_gb.fit(x_train, y_train)
print("results: gradient boosting classifier")
print("r squared: ", model_gb.score(x_test, y_test))
print(classification_report(y_test, model_gb.predict(x_test)))
cm = confusion_matrix(y_test, model_gb.predict(x_test))
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
plt.title("Gradient Boosting", loc="center", fontsize=20)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()
# model discussion
# precision is around 66
# recall is around 66
# accuracy is 66
# still svc is better.

# the last one, Hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRFClassifier

param_grid = {'n_estimators': [100, 80, 60, 55, 51, 45],
              'max_depth': [7, 8],
              'reg_lambda': [0.26, 0.25, 0.2]
              }

model_GSCV = GridSearchCV(XGBRFClassifier(), param_grid, refit=True, verbose=3, n_jobs=-1)
model_GSCV.fit(x_train, y_train)
best_params = model_GSCV.best_params_
# using best params to create and fit model
best_model = XGBRFClassifier(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],
                               reg_lambda=best_params["reg_lambda"])
print("results: hyperparameter tuning")
print("r squared", model_GSCV.score(x_test, y_test))
print( classification_report(y_test, model_GSCV.predict(x_test)))
cm = confusion_matrix(y_test, model_GSCV.predict(x_test))
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
plt.title("Grid Search CV", loc="center", fontsize=20)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()

# model discussion
# precision is around 66
# recall is around 66
# accuracy is 66

# conclusion: to be honest logistic regression has done well with accuracy rate of 65.
# but in the optimization process svc has outperformed all the other models with accuracy rate of 67
#

