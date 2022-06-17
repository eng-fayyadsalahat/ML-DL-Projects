#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv("dataset/CC GENERAL.csv")


# In[11]:


data.head()


# In[12]:


data.describe()


# In[13]:


data.info()


# In[14]:


def null_values(df):
    null_value = df.isnull().sum().sort_values(ascending=False)
    percent_1 = df.isnull().sum() / df.isnull().count() * 100
    percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
    missing_data = pd.concat([null_value, percent_2], axis=1, keys=['Total', '%'])
    print(missing_data)


# In[20]:


null_values(data)


# In[21]:


data["CREDIT_LIMIT"] = data["CREDIT_LIMIT"].fillna(data["CREDIT_LIMIT"].mean())
data["MINIMUM_PAYMENTS"] = data["MINIMUM_PAYMENTS"].fillna(data["MINIMUM_PAYMENTS"].mean()) 


# In[22]:


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


# In[23]:


import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.figure_factory as ff
from plotly.subplots import make_subplots


# In[24]:


pio.renderers.default = "notebook"


# In[25]:


def optimal_bins(df, dt):
    # Shimazaki H. and Shinomoto S.
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
    cmin = C.min()
    idx = np.where(C == cmin)
    idx = idx[0][0]
    return int(N[idx])


# In[26]:


bin_count = optimal_bins(data, "BALANCE")


# In[27]:


fig = px.histogram(data_frame=data, x="BALANCE", nbins=bin_count,histnorm='probability')
fig.show()


# In[28]:


fig = ff.create_distplot([data["BALANCE"]], ["BALANCE"],bin_size=bin_count, show_rug=True, show_curve=True)
fig.update_layout(width= 1000, 
                  height=1000,
                  bargap=0.01)


# In[29]:


c_bincount = optimal_bins(data, "CREDIT_LIMIT")


# In[30]:


fig = px.histogram(data_frame=data, x="CREDIT_LIMIT", nbins=c_bincount, histnorm='probability')
fig.show()


# In[31]:


fig = ff.create_distplot([data["CREDIT_LIMIT"]], ["CREDIT_LIMIT"],bin_size=c_bincount, show_rug=True, show_curve=True)
fig.update_layout(width=1000, 
                  height=1000,
                  bargap=0.01)
fig.show()


# In[32]:


p_bincount = optimal_bins(data, "PURCHASES")


# In[33]:


fig = px.histogram(data_frame=data, x="PURCHASES", nbins=p_bincount)
fig.show()


# In[34]:


fig = ff.create_distplot([data["PURCHASES"]], ["PURCHASES"],bin_size=p_bincount, show_rug=True, show_curve=True)
fig.update_layout(width=1000, 
                  height=1000,
                  bargap=0.01)


# In[35]:


pa_bincount = optimal_bins(data, "PAYMENTS")


# In[36]:


fig = px.histogram(data_frame=data, x="PAYMENTS", nbins=pa_bincount, histnorm='probability')
fig.show()


# In[37]:


fig = ff.create_distplot([data["PAYMENTS"]], ["PAYMENTS"],bin_size=pa_bincount, show_rug=True, show_curve=True)
fig.update_layout(width=1000, 
                  height=1000,
                  bargap=0.01)
fig.show()


# In[38]:


trace0 = go.Box(y=data["BALANCE"], name="BALANCE", boxmean=True)
trace1 = go.Box(y=data["PURCHASES"], name="PURCHASES", boxmean=True)
trace2 = go.Box(y=data["ONEOFF_PURCHASES"], name="ONEOFF_PURCHASES", boxmean=True)
trace3 = go.Box(y=data["INSTALLMENTS_PURCHASES"], name="INSTALLMENTS_PURCHASES", boxmean=True)
trace4 = go.Box(y=data["CASH_ADVANCE"], name="CASH_ADVANCE", boxmean=True)
traces = [trace0, trace1, trace2, trace3, trace4]
layout = go.Layout(title="BoxPlot for Data")
fig_go = go.Figure(data=traces, layout=layout)
fig_go.show()


# In[39]:


trace0 = go.Box(y=data["BALANCE_FREQUENCY"], name="BALANCE_FREQUENCY", boxmean=True)
trace1 = go.Box(y=data["PURCHASES_FREQUENCY"], name="PURCHASES_FREQUENCY", boxmean=True)
trace2 = go.Box(y=data["PURCHASES_INSTALLMENTS_FREQUENCY"], name="PURCHASES_INSTALLMENTS_FREQUENCY", boxmean=True)
trace3 = go.Box(y=data["CASH_ADVANCE_FREQUENCY"], name="CASH_ADVANCE_FREQUENCY", boxmean=True)
trace4 = go.Box(y=data["PRC_FULL_PAYMENT"], name="PRC_FULL_PAYMENT", boxmean=True)
traces = [trace0, trace1, trace2, trace3, trace4]
layout = go.Layout(title="BoxPlot for Data")
fig_go = go.Figure(data=traces, layout=layout)
fig_go.show()


# In[40]:


trace0 = go.Box(y=data["CASH_ADVANCE_TRX"], name="CASH_ADVANCE_TRX", boxmean=True)
trace1 = go.Box(y=data["PURCHASES_TRX"], name="PURCHASES_TRX", boxmean=True)
trace2 = go.Box(y=data["TENURE"], name="TENURE", boxmean=True)
traces = [trace0, trace1, trace2]
layout = go.Layout(title="BoxPlot for Data")
fig_go = go.Figure(data=traces, layout=layout)
fig_go.show()


# In[41]:


trace0 = go.Box(y=data["CREDIT_LIMIT"], name="CREDIT_LIMIT", boxmean=True)
trace1 = go.Box(y=data["PAYMENTS"], name="PAYMENTS", boxmean=True)
trace2 = go.Box(y=data["MINIMUM_PAYMENTS"], name="MINIMUM_PAYMENTS", boxmean=True)
traces = [trace0, trace1, trace2]
layout = go.Layout(title="BoxPlot for Data")
fig_go = go.Figure(data=traces, layout=layout)
fig_go.show()


# # Data Preparation

# In[42]:


# BALANCE
outliers(data,"BALANCE")
# CREDIT_LIMIT
outliers(data,"CREDIT_LIMIT")


# In[43]:


# PURCHASES
outliers(data, "PURCHASES")
# PAYMENTS
outliers(data, "PAYMENTS")


# In[44]:


# BALANCE_FREQUENCY
outliers(data, "BALANCE_FREQUENCY")
# ONEOFF_PURCHASES
outliers(data, "ONEOFF_PURCHASES")


# In[45]:


# INSTALLMENTS_PURCHASES
outliers(data, "INSTALLMENTS_PURCHASES")
# CASH_ADVANCE
outliers(data, "CASH_ADVANCE")


# In[46]:


# PURCHASES_FREQUENCY
outliers(data, "PURCHASES_FREQUENCY")
# ONEOFF_PURCHASES_FREQUENCY
outliers(data, "ONEOFF_PURCHASES_FREQUENCY")


# In[47]:


# MINIMUM_PAYMENTS
outliers(data, "MINIMUM_PAYMENTS")
# PRC_FULL_PAYMENT
outliers(data, "PRC_FULL_PAYMENT")
# TENURE
outliers(data, "TENURE")


# In[48]:


# CASH_ADVANCE_TRX
outliers(data, "CASH_ADVANCE_TRX")
# PURCHASES_TRX
outliers(data, "PURCHASES_TRX")


# In[49]:


# PURCHASES_INSTALLMENTS_FREQUENCY
outliers(data, "PURCHASES_INSTALLMENTS_FREQUENCY")
# CASH_ADVANCE_FREQUENCY
outliers(data, "CASH_ADVANCE_FREQUENCY")


# In[50]:


data["Monthly_PURCHASES"]=data["PURCHASES"]/data["TENURE"]
data["Monthly_CASH_ADVANCE"]=data["CASH_ADVANCE"]/data["TENURE"]


# In[51]:


data['LIMIT_USAGE']=data['BALANCE']/data['CREDIT_LIMIT']


# In[52]:


data["PAYMENTS_MIN"] = data["PAYMENTS"]/data["MINIMUM_PAYMENTS"]


# In[53]:


for d in data:
    data.loc[(data["ONEOFF_PURCHASES"] == 0) & (data["INSTALLMENTS_PURCHASES"] == 0), "PURCHASES_TYPE"] = "none"
    data.loc[(data["ONEOFF_PURCHASES"] > 0) & (
            data["INSTALLMENTS_PURCHASES"] > 0), "PURCHASES_TYPE"] = "oneoff_installment "
    data.loc[(data["ONEOFF_PURCHASES"] > 0) & (data["INSTALLMENTS_PURCHASES"] == 0), "PURCHASES_TYPE"] = "one_off"
    data.loc[
        (data["ONEOFF_PURCHASES"] == 0) & (data["INSTALLMENTS_PURCHASES"] > 0), "PURCHASES_TYPE"] = "istallment"


# In[54]:


data["PURCHASES_TYPE"].head()


# In[55]:


data=data.join(pd.get_dummies(data["PURCHASES_TYPE"]))


# In[56]:


data.info()


# In[57]:


# CUST_ID
df = data
data = data.drop("CUST_ID", axis=1)


# In[58]:


null_values(data)


# # Correlation

# In[59]:


Pearson_corr = data.corr()


# In[60]:


trace = go.Heatmap(z=Pearson_corr.values,
                  x=Pearson_corr.index.values,
                  y=Pearson_corr.columns.values)
traces=[trace]
layout = go.Layout(title="Pearson Correlation" ,width = 1050, height = 900,
    autosize = False)
fig_go = go.Figure(data=traces, layout=layout)
fig_go.show()


# In[61]:


Spearman_corr = data.corr(method="spearman")


# In[62]:


trace = go.Heatmap(z=Spearman_corr.values,
                  x=Spearman_corr.index.values,
                  y=Spearman_corr.columns.values)
traces=[trace]
layout = go.Layout(title="Spearman Correlation" ,width = 1050, height = 900,
    autosize = False)
fig_go = go.Figure(data=traces, layout=layout)
fig_go.show()


# In[63]:


Kendall_corr = data.corr(method= "kendall")


# In[64]:


trace = go.Heatmap(z= Kendall_corr.values,
                  x= Kendall_corr.index.values,
                  y= Kendall_corr.columns.values)
traces=[trace]
layout = go.Layout(title="Kendall Correlation" ,width = 1050, height = 900,
    autosize = False)
fig_go = go.Figure(data=traces, layout=layout)
fig_go.show()


# In[65]:


df_data = data.drop("PURCHASES_TYPE",axis=1)


# In[66]:


df_data.info()


# # Scale Data

# In[67]:


from sklearn.preprocessing import StandardScaler


# In[68]:


scale_data = pd.DataFrame(StandardScaler().fit_transform(df_data), columns= df_data.columns)


# In[69]:


scale_data.head()


# ## Hopikins Statistic

# ### hopikins statistic to test Cluster tendency

# In[70]:


from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan


# In[71]:


def hopkins(X):
    d = X.shape[1]
    # d = len(vars) # columns
    n = len(X)  # rows
    m = int(0.1 * n)  # heuristic from article [1]
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)

    rand_X = sample(range(0, n, 1), m)

    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X, axis=0), np.amax(X, axis=0), d).reshape(1, -1), 2,
                                    return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])

    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0

    return H


# #### test for data before Dimensionality reduction by Principal Component Analysis.

# In[72]:


hopkins(df_data)


# # Clustring

# In[73]:


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics.pairwise import cosine_similarity


# #### cosine similarity.

# In[74]:


data_sim = 1-  cosine_similarity(scale_data)


# In[75]:


pca = PCA(n_components = 0.95, random_state=42)
pca.fit(data_sim)
reduced = pca.transform(data_sim)


# In[76]:


fig = go.Figure()
fig.add_trace(
    go.Line(x= (pca.explained_variance_ratio_).tolist(),
            y= list(range(0,6)) )
)
fig.update_layout(
    title="Principal Component Analysis",
    yaxis_title="Number of Components",
    xaxis_title="Explained Variance Ratio",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)

fig.show()


# In[77]:


df_pca = pd.DataFrame({"PC1":pca.components_[0],"PC2":pca.components_[1],"PC3":pca.components_[2],"PC4":pca.components_[3],"PC5":pca.components_[4],"PC6":pca.components_[5]})
df_pca.head()


# In[78]:


pca_dict= {'Feature':list(scale_data.columns),"PC1":pca.components_[0],"PC2":pca.components_[1],"PC3":pca.components_[2],"PC4":pca.components_[3],"PC5":pca.components_[4],"PC6":pca.components_[5]}
pcas_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in pca_dict.items()]))
pcas_df.head(24)


# In[79]:


fig = px.scatter(pcas_df, x = "Feature", y =["PC1","PC2","PC3","PC4","PC5","PC6"], labels={"x":"Feature", "y":"PCA"})
fig.show()


# In[80]:


hopkins(df_pca)


# ## IncrementalPCA

# In[81]:


pca_inc = IncrementalPCA(n_components=6)
reduced_inc = pca_inc.fit_transform(data_sim)


# In[82]:


fig = go.Figure()
fig.add_trace(
    go.Line(x= (pca_inc.explained_variance_ratio_).tolist(),
            y= list(range(0,6)) )
)
fig.update_layout(
    title=" Incremental Principal Component Analysis",
    yaxis_title="Number of Components",
    xaxis_title="Explained Variance Ratio",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)

fig.show()


# In[83]:


df_pca_inc = pd.DataFrame({"PC1":pca_inc.components_[0],"PC2":pca_inc.components_[1],"PC3":pca_inc.components_[2],
                           "PC4":pca_inc.components_[3],"PC5":pca_inc.components_[4],"PC6":pca_inc.components_[5]})
df_pca_inc.head()


# In[84]:


pca_inc_dict= {'Feature':list(scale_data.columns), "PC1":pca_inc.components_[0],"PC2":pca_inc.components_[1],"PC3":pca_inc.components_[2],
                           "PC4":pca_inc.components_[3],"PC5":pca_inc.components_[4],"PC6":pca_inc.components_[5]}
pcas_inc_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in pca_inc_dict.items()]))
pcas_inc_df.head(24)


# In[85]:


fig = px.scatter(pcas_inc_df, x = "Feature", y =["PC1","PC2","PC3","PC4","PC5","PC6"])
fig.show()


# In[86]:


hopkins(df_pca_inc)


# # Elbow law

# In[87]:


cost=[]
for n_clusters in range(2,16):
    kmean= KMeans(n_clusters=n_clusters)
    kmean.fit(df_pca_inc)
    cost.append(kmean.inertia_)


# In[88]:


fig = px.line(x= cost, y= list(range(2,16)), title="The Elbow method for optimal K", labels={"y":"Number of Cluster", "x":"WCSS"})
fig.show()


# # Silhouette Score

# In[89]:


from sklearn.metrics import silhouette_score


# In[90]:


sse_incpca = {}
for n_clusters in range(2,16):
    clusterer = KMeans(n_clusters=n_clusters)
    preds = clusterer.fit_predict(df_pca_inc)
    centers = clusterer.cluster_centers_
    sse_incpca[n_clusters] = silhouette_score(df_pca_inc, clusterer.labels_)
    score = silhouette_score(df_pca_inc, preds)
    print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))


# In[91]:


fig = px.line(x= sse_incpca.values(), y= sse_incpca.keys(), title="The Silhouette Score for optimal K", labels={"y":"Number of Cluster", "x":"Score"})
fig.show()


# # K-MEANS Algorithm

# In[92]:


k_means = KMeans(n_clusters=4, max_iter=100000, random_state =42)
k_means.fit(df_pca_inc)
data_kmeans = k_means.transform(df_pca_inc)


# In[93]:


clusters = pd.concat([df_pca_inc, pd.DataFrame({"ClusterID":k_means.labels_})], axis=1)
clusters.tail()


# In[94]:


clusters.ClusterID.value_counts()


# In[95]:


hopkins(clusters)


# In[96]:


from sklearn import metrics
metrics.calinski_harabasz_score(df_pca_inc, k_means.labels_)


# In[97]:


silhouette_score(df_pca_inc, k_means.labels_)


# In[98]:


cluster0 = clusters[clusters["ClusterID"] == 0]
cluster1 = clusters[clusters["ClusterID"] == 1]
cluster2 = clusters[clusters["ClusterID"] == 2]
cluster3 = clusters[clusters["ClusterID"] == 3]


# In[99]:


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
trace1 = go.Scatter3d(
                    x = cluster0["PC1"],
                    y =  cluster0["PC2"],
                    z = cluster0["ClusterID"],
                    mode = "markers",
                    name = "Cluster 0",
                    marker = dict(color = "red"),
                    text = None)

#trace2 is for 'Cluster 1'
trace2 = go.Scatter3d(
                    x = cluster1["PC1"],
                    y = cluster1["PC2"],
                    z = cluster1["ClusterID"],
                    mode = "markers",
                    name = "Cluster 1",
                    marker = dict(color = "black"),
                    text = None)

#trace3 is for 'Cluster 2'
trace3 = go.Scatter3d(
                    x = cluster2["PC1"],
                    y = cluster2["PC2"],
                    z = cluster2["ClusterID"],
                    mode = "markers",
                    name = "Cluster 2",
                    marker = dict(color = "white"),
                    text = None)
#trace4 is for 'Cluster 3'
trace4 = go.Scatter3d(
                    x = cluster3["PC1"],
                    y = cluster3["PC2"],
                    z = cluster3["ClusterID"],
                    mode = "markers",
                    name = "Cluster 3",
                    marker = dict(color = "yellow"),
                    text = None)


data = [trace1, trace2, trace3, trace4]

title = "Visualizing Clusters in Three Dimensions Using PCA"

layout = dict(title = title,
              xaxis= dict(title= 'PC1',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'PC2',ticklen= 5,zeroline= False)
             )

fig = dict(data = data, layout = layout)

iplot(fig)


# In[100]:


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
trace1 = go.Scatter(
                    x = cluster0["PC1"],
                    y =  cluster0["PC2"],
                   
                    mode = "markers",
                    name = "Cluster 0",
                    marker = dict(color = "red"),
                    text = None)

#trace2 is for 'Cluster 1'
trace2 = go.Scatter(
                    x = cluster1["PC1"],
                    y = cluster1["PC2"],
                    
                    mode = "markers",
                    name = "Cluster 1",
                    marker = dict(color = "black"),
                    text = None)

#trace3 is for 'Cluster 2'
trace3 = go.Scatter(
                    x = cluster2["PC1"],
                    y = cluster2["PC2"],
                    
                    mode = "markers",
                    name = "Cluster 2",
                    marker = dict(color = "green"),
                    text = None)
#trace4 is for 'Cluster 3'
trace4 = go.Scatter(
                    x = cluster3["PC1"],
                    y = cluster3["PC2"],
                    
                    mode = "markers",
                    name = "Cluster 3",
                    marker = dict(color = "yellow"),
                    text = None)


data = [trace1, trace2, trace3, trace4]

title = "Visualizing Clusters in Two Dimensions Using PCA"

layout = dict(title = title,
              xaxis= dict(title= 'PC1',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'PC2',ticklen= 5,zeroline= False)
             )

fig = dict(data = data, layout = layout)

iplot(fig)

