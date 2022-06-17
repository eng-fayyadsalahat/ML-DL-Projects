#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 


# In[ ]:


train =pd.read_csv("dataset/train.csv")


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio 


# In[ ]:


pio.renderers.default = "notebook"


# In[ ]:


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


# In[ ]:


def null_values(df):
    null_value = df.isnull().sum().sort_values(ascending=False)
    percent_1 = df.isnull().sum() / df.isnull().count() * 100
    percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
    missing_data = pd.concat([null_value, percent_2], axis=1, keys=['Total', '%'])
    print(missing_data)


# In[ ]:


null_values(train)


# In[ ]:


train = train.ffill()
train = train.bfill()


# In[ ]:


train = train.drop(columns=["Id", "PoolQC", "Fence","Alley","MiscFeature"], axis=1)


# In[ ]:


from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


# In[ ]:


def cat_to_num(df,dt):
    enc = OrdinalEncoder()
    df[[dt]] = enc.fit_transform(df[[dt]])
    df[dt]=df[dt].astype("int64")
    outliers(df,dt)


# In[ ]:


for item in (train.loc[:, train.dtypes == np.object].columns):
    cat_to_num(train,item)


# In[ ]:


train = train.astype("int64")


# ## Corrilation

# In[ ]:


corr = train.corr()


# In[ ]:


corr["SalePrice"]


# In[ ]:


columns_toremove = list(corr.index[ corr["SalePrice"]<0])


# In[ ]:


train = train.drop(columns= columns_toremove, axis=1)


# In[ ]:


trace = go.Heatmap(z=corr.values,
                  x=corr.index.values,
                  y=corr.columns.values)
traces=[trace]
layout = go.Layout(title=" Correlation" ,width = 1050, height = 900,
    autosize = False)
fig_go = go.Figure(data=traces, layout=layout)
fig_go.show()


# In[ ]:


train.shape


# In[ ]:


train.shape


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


# In[ ]:


numeric_coloumns = list(train.columns)
pipeline = ColumnTransformer([
    ("Standred", StandardScaler(), numeric_coloumns,)
])
scaled_data = pd.DataFrame(pipeline.fit_transform(train), columns=list(train.columns))


# In[ ]:


label = train["SalePrice"]
scaled_data = scaled_data.drop("SalePrice", axis=1)


# In[ ]:


from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split


# In[ ]:


train_x, test_x, train_y, test_y = train_test_split(scaled_data, label, test_size=0.2, random_state=42)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


regressor = RandomForestRegressor(n_estimators = 100, random_state = 42)
regressor.fit(train_x, train_y)


# In[ ]:


y_pred = regressor.predict(test_x)


# In[ ]:


y_pred


# In[ ]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(test_y, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(test_y, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, y_pred)))


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

estimator = RandomForestRegressor(max_features='auto', oob_score=True, random_state=42)
param_grid = { 
        "n_estimators"      : [200, 400, 700],
        "min_samples_split" : [2,4,8,10],
            }

grid = GridSearchCV(estimator, param_grid, cv=10, n_jobs=6)

grid.fit(train_x, train_y)


# In[ ]:


print(" Results from Grid Search ")
print("\n The best estimator across ALL searched params:\n", grid.best_estimator_)
print("\n The best score across ALL searched params:\n", grid.best_score_)
print("\n The best parameters across ALL searched params:\n", grid.best_params_)


# In[ ]:


model = RandomForestRegressor(n_estimators=1000, oob_score=True, random_state=42)
model.fit(train_x, train_y)


# In[ ]:


model_pred = model.predict(test_x)


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(test_y, model_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(test_y, model_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, model_pred)))


# ## Test

# In[ ]:


test = pd.read_csv("dataset/test.csv")


# In[ ]:


test.info()


# In[ ]:


test = test.drop(columns=["PoolQC", "Fence","Alley","MiscFeature"], axis=1)


# In[ ]:


test = test.drop(columns=columns_toremove, axis=1)


# In[ ]:


test=test.ffill()
test=test.bfill()


# In[ ]:


for item in (test.loc[:, test.dtypes == np.object].columns):
    cat_to_num(test,item)


# In[ ]:


test = test.astype("int64")


# In[ ]:


test_id = test["Id"].astype("int32")
test = test.drop("Id",axis=1)


# In[ ]:


pred = model.predict(test)


# In[ ]:


pred

