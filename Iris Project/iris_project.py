#!/usr/bin/env python
# coding: utf-8

# In[106]:


import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn import utils
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from tabulate import tabulate
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score, recall_score


# In[43]:


data = pd.read_csv("dataset/IRIS.csv")


# In[44]:


data.head()


# In[45]:


data.describe()


# In[46]:


data.info()


# In[47]:


def outliers(df, dt):
    sorted(df[dt])
    Q1 = df[dt].quantile(0.25)
    Q3 = df[dt].quantile(0.75)
    IQR = Q3 - Q1
    print("Column:", dt)
    upper_val = (Q3 + (1.5 * IQR))
    lower_val = Q1 - (1.5 * IQR)
    count = len(df[(df[dt] > upper_val) | (df[dt] < lower_val)])
    df.replace(df[(df[dt] > upper_val) | (df[dt] < lower_val)].index, df[dt].mean(), inplace=True)
    print("Count of Item Replace:", count)
    print("Outliers ratio:", count / len(df[dt]))


# In[48]:


def optimal_bins(df, dt):
    # Shimazaki H. and Shinomoto S.
    data_max = max(df[dt])  # lower end of data
    data_min = min(df[dt])  # upper end of data
    n_min = 2  # Minimum number of bins Ideal value = 2
    n_max = int(pd.Series.count(df[dt]))  # Maximum number of bins  Ideal value =200
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
    cmin = C.min()
    idx = np.where(C == cmin)
    idx = idx[0][0]
    return int(N[idx])


# In[49]:


pio.renderers.default = "notebook"


# In[50]:


bin_count = optimal_bins(data, "sepal_length")
fig = px.histogram(data_frame=data, x="sepal_length", nbins=bin_count)
fig.show()


# In[51]:


trace0 = go.Box(y=data["sepal_length"], name="sepal_length", boxmean=True)
trace1 = go.Box(y=data["sepal_width"], name="sepal_width", boxmean=True)
trace2 = go.Box(y=data["petal_length"], name="petal_length", boxmean=True)
trace3 = go.Box(y=data["petal_width"], name="petal_width", boxmean=True)
traces = [trace0, trace1, trace2, trace3]
layout = go.Layout(title="BoxPlot for Data")
fig_go = go.Figure(data=traces, layout=layout)
fig_go.show()


# In[52]:


fig_sac = px.scatter(data_frame=data, x="sepal_width", y="sepal_length", color="species",
                     size='petal_length', hover_data=['petal_width'])
fig_sac.show()


# In[53]:


feature = ["sepal_width", "sepal_length", "petal_width", "petal_length"]

fig_scm = px.scatter_matrix(
    data,
    dimensions=feature,
    color="species"
)
fig_scm.update_traces(diagonal_visible=False)
fig_scm.show()


# In[54]:


outliers(data, "sepal_width")


# In[55]:


le = LabelEncoder()
data[["species_encode"]] = le.fit_transform(data[["species"]])
data["species_encode"] = data["species_encode"].ravel()


# In[56]:


data = data.drop("species", axis=1)


# In[57]:


##Correlation


# In[58]:


corr = data.corr()
corr["species_encode"].sort_values(ascending=False)


# In[59]:


fig_heatmap = px.imshow(data)
fig_heatmap.show()


# In[60]:


# Standerd Scaler


# In[61]:


numeric_coloumns = ["sepal_width", "sepal_length",
                    "petal_width", "petal_length"]
pipeline = ColumnTransformer([
    ("Standred", StandardScaler(), numeric_coloumns,)
])


# In[62]:


scaled_data = pd.DataFrame(pipeline.fit_transform(data), columns=["sepal_width", "sepal_length",
                                                                  "petal_width", "petal_length"])
train_x, test_x, train_y, test_y = train_test_split(scaled_data, data["species_encode"], test_size=0.2, random_state=42)


# In[98]:


# Random Forest
random_forest = RandomForestClassifier(oob_score=True)
random_forest.fit(train_x, train_y)
random_forest_prediction = random_forest.predict(test_x)
random_forest.score(train_x, train_y)
random_forest_accuracy = round(random_forest.score(train_x, train_y) * 100, 3)


# In[64]:


# KNN
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(train_x, train_y)
knn_prediction = knn.predict(test_x)
knn_accuracy = round(knn.score(train_x, train_y) * 100, 3)


# In[65]:


# Gaussian Naive Bayes
gaussian_naive_bayes = GaussianNB()
gaussian_naive_bayes.fit(train_x, train_y)
gaussian_naive_bayes_prediction = gaussian_naive_bayes.predict(test_x)
gaussian_naive_bayes_accuracy = round(gaussian_naive_bayes.score(train_x, train_y) * 100, 3)


# In[66]:


# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_x, train_y)
decision_tree_prediction = decision_tree.predict(test_x)
decision_tree_accuracy = round(decision_tree.score(train_x, train_y) * 100, 3)


# In[67]:


# Stochastic Gradient Descent (SGD) ovo//
sgd = SGDClassifier()
sgd.fit(train_x, train_y)
sgd_prediction = sgd.predict(test_x)
sgd.score(train_x, train_y)
sgd_accuracy = round(sgd.score(train_x, train_y) * 100, 3)


# In[68]:


# Perceptron 
perceptron = Perceptron()
perceptron.fit(train_x, train_y)
perceptron_prediction = perceptron.predict(test_x)
perceptron_accuracy = round(perceptron.score(train_x, train_y) * 100, 3)


# In[73]:


# Linear Support Vector Machine
linear_svc = LinearSVC()
linear_svc.fit(train_x, train_y)
linear_svc_prediction = linear_svc.predict(test_x)
linear_svc_accuracy = round(linear_svc.score(train_x, train_y) * 100, 3)


# In[99]:


Algorithms = ["LVM", "KNN", "RFC", "GNB", "Perceptron", "SGD", "DTC"]
Scores = [linear_svc_accuracy, knn_accuracy, random_forest_accuracy,
          gaussian_naive_bayes_accuracy, perceptron_accuracy, sgd_accuracy, decision_tree_accuracy]
Accuracy = [Algorithms, Scores]
print(tabulate(Accuracy, tablefmt="pipe"))


# In[80]:


# OvO or OvR strategy


# In[81]:


# OvO for SGD
ovo_sgd = OneVsOneClassifier(SGDClassifier(max_iter=150))
ovo_sgd.fit(train_x, train_y)
ovo_sgd_prediction = ovo_sgd.predict(test_x)
ovo_sgd_accuracy = round(ovo_sgd.score(train_x, train_y) * 100, 3)


# In[82]:


# OvR for SGD
ovr_sgd = OneVsRestClassifier(SGDClassifier(max_iter=150))
ovr_sgd.fit(train_x, train_y)
ovr_sgd_prediction = ovr_sgd.predict(test_x)
ovr_sgd_accuracy = round(ovr_sgd.score(train_x, train_y) * 100, 3)


# In[83]:


# OvO for Perceptron
ovo_perceptron = OneVsRestClassifier(Perceptron())
ovo_perceptron.fit(train_x, train_y)
ovo_perceptron_prediction = ovo_perceptron.predict(test_x)
ovo_perceptron_accuracy = round(ovo_perceptron.score(train_x, train_y) * 100, 3)


# In[84]:


# OvR for Perceptron
ovr_perceptron = OneVsRestClassifier(Perceptron())
ovr_perceptron.fit(train_x, train_y)
ovr_perceptron_prediction = ovr_perceptron.predict(test_x)
ovr_perceptron_accuracy = round(ovr_perceptron.score(train_x, train_y) * 100, 3)


# In[85]:


# OvO for Linear Support Vector Machine
ovo_lsv = OneVsRestClassifier(LinearSVC())
ovo_lsv.fit(train_x, train_y)
ovo_lsv_prediction = ovo_lsv.predict(test_x)
ovo_lsv_accuracy = round(ovo_lsv.score(train_x, train_y) * 100, 3)


# In[86]:


# OvR for Linear Support Vector Machine
ovr_lsv = OneVsRestClassifier(LinearSVC())
ovr_lsv.fit(train_x, train_y)
ovr_lsv_prediction = ovr_lsv.predict(test_x)
ovr_lsv_accuracy = round(ovr_lsv.score(train_x, train_y) * 100, 3)


# In[87]:


Algorithms_Strategy = ["OVO for SGD", "OVR for SGD", "OVO for  Perceptron", "OVR for Perceptron",
                       "OVO for LVM", "OVR for LVM"]
Scores_Strategy = [ovo_sgd_accuracy, ovr_sgd_accuracy, ovo_perceptron_accuracy, ovr_perceptron_accuracy,
                   ovo_lsv_accuracy, ovr_lsv_accuracy]
Accuracy_Strategy = [Algorithms_Strategy, Scores_Strategy]
print(tabulate(Accuracy_Strategy, tablefmt="pipe"))


# In[97]:


# Cross Validations Random Forest Classifier
rf = RandomForestClassifier()
scores = cross_val_score(rf, train_x, train_y, cv=10, scoring="accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# In[96]:


# Cross Validations Decision Tree
dt = DecisionTreeClassifier()
scores = cross_val_score(dt, train_x, train_y, cv=10, scoring="accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# In[101]:


# Out Of Bag Samples
print("oob score:", round(random_forest.oob_score_, 5)*100, "%")


# In[90]:


# Grid Search
param_grid = {"criterion": ["gini", "entropy"], "min_samples_leaf": [1, 5, 10, 25, 40],
              "min_samples_split": [2, 4, 10, 12, 16, 18, 24, 30],
              "n_estimators": [100, 400, 700]}
rf_g = RandomForestClassifier(max_features='auto', oob_score=True, random_state=42)
grid_rf = GridSearchCV(estimator=rf_g, param_grid=param_grid, n_jobs=6, cv=5)
grid_rf.fit(train_x, train_y)

print(" Results from Grid Search ")
print("\n The best estimator across ALL searched params:\n", grid_rf.best_estimator_)
print("\n The best score across ALL searched params:\n", grid_rf.best_score_)
print("\n The best parameters across ALL searched params:\n", grid_rf.best_params_)


# In[116]:


new_random_forest_classifier = RandomForestClassifier(min_samples_split=4, n_estimators=400, oob_score=True,
                                                      random_state=42)

new_random_forest_classifier.fit(train_x,train_y)
new_random_forest_classifier_accuracy = round(new_random_forest_classifier.score(train_x, train_y) * 100, 3)
print(new_random_forest_classifier_accuracy)


# In[104]:


# Out Of Bag Samples
print("oob score:", round(new_random_forest_classifier.oob_score_, 5)*100, "%")


# In[92]:


# Classification Report
# Confusion Matrix
predictions = cross_val_predict(new_random_forest_classifier, train_x, train_y, cv=10)
print("Confusion Matrix:\n", confusion_matrix(train_y, predictions))
print("Classification Report:\n", classification_report(train_y, predictions))


# In[122]:


from sklearn.metrics import roc_auc_score
y_scores = new_random_forest_classifier.predict_proba(train_x)
r_a_score = roc_auc_score(train_y, y_scores)
print("ROC-AUC-Score:", r_a_score)


# In[ ]:




