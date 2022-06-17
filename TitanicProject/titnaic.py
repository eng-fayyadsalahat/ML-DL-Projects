#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Titanic Project


# In[2]:


# Import Libraries 


# In[60]:


import numpy as np
import pandas as pd
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats 
from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


def outliers(df, dt):
    sorted(df[dt])
    Q1 = df[dt].quantile(0.25)
    Q3 = df[dt].quantile(0.75)
    IQR = Q3 - Q1
    print("Column:", dt)
    print("Old Shape", df.shape)
    upper_val = (Q3 + (1.5 * IQR))
    lower_val = Q1 - (1.5 * IQR)
    count = len(df[(df[dt] > upper_val) | (df[dt] < lower_val)])
    df.drop(df[(df[dt] > upper_val) | (df[dt] < lower_val)].index, inplace=True)
    print("New Shape", df.shape)
    print("Count of Item Removed:", count)
    print("Outliers ratio:", count / len(df[dt]))


def null_values(df):
    null_value = df.isnull().sum().sort_values(ascending=False)
    percent_1 = df.isnull().sum() / df.isnull().count() * 100
    percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
    missing_data = pd.concat([null_value, percent_2], axis=1, keys=['Total', '%'])
    print(missing_data)


# In[4]:


train_data = pd.read_csv("dataset/titanic_train.csv")
test_data = pd.read_csv("dataset/titanic_test.csv")


# In[5]:


# display first 5 row in train data
train_data.head()


# In[6]:


# display description of train data
train_data.describe()


# In[7]:


# display information about train data
train_data.info()


# In[8]:


# display first 5 row in test data
test_data.head()


# In[9]:


# display description of test data
test_data.describe()


# In[10]:


# display information about test data
test_data.info()


# In[11]:


# check for the null train_data
null_values(train_data)


# In[12]:


# check for the null test data
null_values(test_data)


# In[13]:


# Correlation for train data
corr = train_data.corr()
corr["Survived"].sort_values(ascending=False)


# In[14]:


# Data Visualization 


# In[15]:


plt.style.use("ggplot")
sns.set_style("darkgrid")


# In[16]:


# histogram


# In[17]:


# Shimazaki H. and Shinomoto S. rule to calculate optimal bins in histogram
def optimal_bins(df, dt):
    data_max = max(df[dt])  # lower end of data
    data_min = min(df[dt])  # upper end of data
    n_min = 2  # Minimum number of bins Ideal value = 2
    n_max = 200  # Maximum number of bins  Ideal value =200
    n_shift = 30  # number of shifts Ideal value = 30
    N = np.array(range(n_min, n_max))
    D = float(data_max - data_min) / N  # Bin width vector
    Cs = np.zeros((len(D), n_shift))  # Cost function vector
    for i in range(np.size(N)):
        shift = np.linspace(0, D[i], n_shift)
        for j in range(n_shift):
            edges = np.linspace(data_min + shift[j] - D[i] / 2, data_max + shift[j] - D[i] / 2,
                                N[i] + 1)  # shift the Bin edges
            binindex = np.digitize(df[dt], edges)  # Find binindex of each data point
            ki = np.bincount(binindex)[1:N[i] + 1]  # Find number of points in each bin
            k = np.mean(ki)  # Mean of event count
            v = sum((ki - k) ** 2) / N[i]  # Variance of event count
            Cs[i, j] += (2 * k - v) / ((D[i]) ** 2)  # The cost Function
    C = Cs.mean(1)
    # Optimal Bin Size Selection
    loc = np.argwhere(Cs == Cs.min())[0]
    cmin = C.min()
    idx = np.where(C == cmin)
    idx = idx[0][0]
    optD = D[idx]
    return N[idx]


# In[18]:


# histogram for Age Series 
bin_count = optimal_bins(train_data,"Age")
bin_count
# calculate optimal bins numbers and display it
plt.hist(train_data["Age"], edgecolor='black', bins=bin_count, color="steelblue")
plt.xlabel("Age")
plt.ylabel("Count")


# In[19]:


# target train_data
# Survived is our target train_data


# In[20]:


sns.countplot(x="Survived", data=train_data, hue="Sex")


# In[21]:


sns.countplot(x="Survived", data=train_data, hue="Pclass")


# In[22]:


# show boxplot
# Create a figure and a subplots, with size of figure 15X4
# A box plot shows the distribution of quantitative data in a way that facilitates comparisons between variables
# or across levels of a categorical variable
fig, ax = plt.subplots(1, 4, figsize=(15, 4))
sns.boxplot(data=train_data, x="Age", color="#ce181f", ax=ax[0], showmeans=True)
sns.boxplot(data=train_data, x="SibSp", color="#232f51", ax=ax[1], showmeans=True)
sns.boxplot(data=train_data, x="Parch", color="#3b3742", ax=ax[2], showmeans=True)
sns.boxplot(data=train_data, x="Fare", color="#70dc88", ax=ax[3], showmeans=True)


# In[62]:


# Jointplot
r, p = stats.pearsonr(train_data["Age"], train_data["Survived"])
j = sns.jointplot(x="Age", y="Survived", kind="kde", fill=True,
             thresh=0, data=train_data, color="blue")
phantom, = j.ax_joint.plot([], [], linestyle="", alpha=0)
j.ax_joint.legend([phantom],["p={:f},r={:f}".format(r,p)])


# In[65]:


r, p = stats.pearsonr(train_data["Fare"], train_data["Survived"])
j = sns.jointplot(x="Fare",y="Survived",data= train_data, color='steelblue')
phantom, = j.ax_joint.plot([], [], linestyle="", alpha=0)
j.ax_joint.legend([phantom],["p={:f},r={:f}".format(r,p)])


# In[23]:


# Analyze Data and Preparing  Data


# In[24]:


# Correlation
corr = train_data.corr()
print(corr["Survived"].sort_values(ascending=False))


# In[25]:


# PassengerId values
# we don't need this feature, so we go to drop this
train_data = train_data.drop("PassengerId", axis=1)
test_data = test_data.drop("PassengerId", axis=1)


# In[26]:


# Name values
# we don't need this feature, so we go to drop this
train_data = train_data.drop("Name", axis=1)
test_data = test_data.drop("Name", axis=1)


# In[27]:


# Sex values
# Convert from categorical to numerical
enc = OrdinalEncoder()
train_data[["Gender"]] = enc.fit_transform(train_data[["Sex"]])
test_data[["Gender"]] = enc.fit_transform(test_data[["Sex"]])
train_data = train_data.drop("Sex", axis=1)
test_data = test_data.drop("Sex", axis=1)


# In[28]:


# Age values
# Fill null value
train_data["Age"] = train_data["Age"].fillna(method="ffill")
# Cast data from type float to int
train_data["Age"] = train_data["Age"].astype(int)
# Drop outliers data
outliers(train_data, "Age")
test_data["Age"] = test_data["Age"].fillna(method="ffill")
test_data["Age"] = test_data["Age"].astype(int)
outliers(test_data, "Age")


# In[29]:


# Categorizing every age into a groups
data_set = [train_data, test_data]
for data in data_set:
    data.loc[data["Age"] <= 12, "Age"] = 0  # child passengers
    data.loc[(data["Age"] > 12) & (data["Age"] <= 20), "Age"] = 1  # teenage passengers
    data.loc[(data["Age"] > 20) & (data["Age"] <= 30), "Age"] = 2  # young passengers
    data.loc[(data["Age"] > 30) & (data["Age"] <= 40), "Age"] = 3  # man passengers
    data.loc[(data["Age"] > 40) & (data["Age"] <= 50), "Age"] = 4  # adult passengers
    data.loc[(data["Age"] > 50) & (data["Age"] <= 60), "Age"] = 5  # old passengers
    data.loc[(data["Age"] > 60), "Age"] = 6  # very old passengers


# In[30]:


# check Relative between passengers and Relatives
train_data["Relatives"] = train_data["SibSp"] + train_data["Parch"]
train_data.loc[train_data["Relatives"] > 0, "not_singular"] = 0
train_data.loc[train_data["Relatives"] == 0, "not_singular"] = 1
train_data["not_singular"] = train_data["not_singular"].astype(int)

test_data["Relatives"] = test_data["SibSp"] + test_data["Parch"]
test_data.loc[test_data["Relatives"] > 0, "not_singular"] = 0
test_data.loc[test_data["Relatives"] == 0, "not_singular"] = 1
test_data["not_singular"] = test_data["not_singular"].astype(int)


# In[31]:


# Ticket values
# drop this because we have many unique value
train_data = train_data.drop("Ticket", axis=1)
test_data = test_data.drop("Ticket", axis=1)


# In[32]:


# Fare values
# Fill null value
train_data["Fare"] = train_data["Fare"].fillna(method="ffill")
test_data["Fare"] = test_data["Fare"].fillna(method="ffill")
# Drop outliers data
outliers(train_data, "Fare")
outliers(test_data, "Fare")


# In[33]:


# Cabin values
# drop this becuse we have many null value
train_data = train_data.drop("Cabin", axis=1)
test_data = test_data.drop("Cabin", axis=1)


# In[34]:


# Embarked values
# Fill null values
train_data["Embarked"] = train_data["Embarked"].fillna(method="ffill")
test_data["Embarked"] = test_data["Embarked"].fillna(method="ffill")
# Convert from categorical to numerical
enc = OrdinalEncoder()
train_data[["Embarked_encode"]] = enc.fit_transform(train_data[["Embarked"]])
test_data[["Embarked_encode"]] = enc.fit_transform(test_data[["Embarked"]])
train_data = train_data.drop("Embarked", axis=1)
test_data = test_data.drop("Embarked", axis=1)


# In[35]:


train_data.head()


# In[36]:


test_data.head()


# In[37]:


# Building Machine Learning Models


# In[38]:


# Standardization
numeric_coloumns = ["Pclass", "Age", "SibSp", "Parch",
                    "Fare", "Gender", "Relatives", "not_singular", "Embarked_encode"]
pipeline = ColumnTransformer([
    ("num", StandardScaler(), numeric_coloumns,)
])
train_data_scaled = pd.DataFrame(pipeline.fit_transform(train_data), columns=["Pclass", "Age", "SibSp", "Parch",
                                                                              "Fare", "Gender", "Relatives",
                                                                              "not_singular", "Embarked_encode"])
test_data_scaled = pd.DataFrame(pipeline.fit_transform(test_data), columns=["Pclass", "Age", "SibSp", "Parch",
                                                                            "Fare", "Gender", "Relatives",
                                                                            "not_singular", "Embarked_encode"])


# In[39]:


train_y = train_data["Survived"]
train_x = train_data_scaled
test_x = test_data_scaled


# In[40]:


# Logistic Regression
logistic_regression = LogisticRegression()
logistic_regression.fit(train_x, train_y)
logistic_regression_accuracy = round(logistic_regression.score(train_x, train_y) * 100, 3)


# In[43]:


# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(train_x, train_y)
random_forest_accuracy = round(random_forest.score(train_x, train_y) * 100, 3)


# In[44]:


# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_x, train_y)
knn_accuracy = round(knn.score(train_x, train_y) * 100, 3)


# In[45]:


# Gaussian Naive Bayes
gaussian_naive_bayes = GaussianNB()
gaussian_naive_bayes.fit(train_x, train_y)
gaussian_naive_bayes_accuracy = round(gaussian_naive_bayes.score(train_x, train_y) * 100, 3)


# In[46]:


# Perceptron
perceptron = Perceptron(max_iter=5)
perceptron.fit(train_x, train_y)
perceptron_accuracy = round(perceptron.score(train_x, train_y) * 100, 3)


# In[47]:


# Linear Support Vector Machine:
linear_svc = LinearSVC()
linear_svc.fit(train_x, train_y)
linear_svc_accuracy = round(linear_svc.score(train_x, train_y) * 100, 3)


# In[48]:


# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_x, train_y)
decision_tree_accuracy = round(decision_tree.score(train_x, train_y) * 100, 3)


# In[49]:


# Stochastic Gradient Descent (SGD)
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(train_x, train_y)
sgd_accuracy = round(sgd.score(train_x, train_y) * 100, 3)


# In[50]:


# Display accuracy of Algorithms
Algorithms = ["Linear Support Vector Machines", "KNN", "Logistic Regression",
              "Random Forest Classifier", "Gaussian Naive Bayes", "Perceptron",
              "Stochastic Gradient Decent",
              "Decision Tree"]
Scores = [linear_svc_accuracy, knn_accuracy, logistic_regression_accuracy, random_forest_accuracy,
          gaussian_naive_bayes_accuracy, perceptron_accuracy, sgd_accuracy, decision_tree_accuracy]
Accuracy = [Algorithms, Scores]
print(tabulate(Accuracy))


# In[51]:


# Cross Validations Random Forest Classifier
rf = RandomForestClassifier()
scores = cross_val_score(rf, train_x, train_y, cv=10, scoring="accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# In[52]:


# Cross Validations Decision Tree
dt = DecisionTreeClassifier()
scores = cross_val_score(dt, train_x, train_y, cv=10, scoring="accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# In[73]:


# Grid Search
param_grid = {"criterion": ["gini", "entropy"], "min_samples_leaf": [1, 5, 10, 25, 40],
              "min_samples_split": [2, 4, 10, 12, 16, 18, 24, 30],
              "n_estimators": [100, 400, 700]}

rf_g = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score=True, random_state=42)
grid_rf = GridSearchCV(estimator=rf_g, param_grid=param_grid, n_jobs=5, cv=8)
grid_rf.fit(train_x, train_y)
print(" Results from Grid Search ")
print("\n The best estimator across ALL searched params:\n", grid_rf.best_estimator_)
print("\n The best score across ALL searched params:\n", grid_rf.best_score_)
print("\n The best parameters across ALL searched params:\n", grid_rf.best_params_)


# In[77]:


# RFC with new parm
random_forest = RandomForestClassifier(min_samples_split=16, n_estimators=100, oob_score=True,
                                       random_state=42, criterion="gini", min_samples_leaf=1,
                                       max_features="auto", n_jobs=5)
random_forest.fit(train_x,train_y)
random_forest_accuracy = round(random_forest.score(train_x, train_y) * 100, 3)
random_forest_accuracy


# In[78]:


# Classification Report
# Confusion Matrix
predictions = cross_val_predict(random_forest, train_x, train_y, cv=8)
print("Confusion Matrix:\n", confusion_matrix(train_y, predictions))
print("Classification Report:\n", classification_report(train_y, predictions))


# In[ ]:




