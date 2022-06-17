
import numpy as np 
import pandas as pd



import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


balanced_train = pd.read_csv("emnist-balanced-train.csv")
balanced_test = pd.read_csv("emnist-balanced-test.csv")
mapp = pd.read_csv("emnist-balanced-mapping.txt", delimiter = ' ', index_col=0, header=None, squeeze=True)


print("Train: %s, Test: %s, Map: %s" %(balanced_train.shape, balanced_test.shape, mapp.shape))



balanced_train.head()


balanced_test.head()


# label of data
balanced_train["45"]



# label of data
balanced_test["41"]





classes = len(balanced_train['45'].value_counts())
print('number of classes: ', classes)





train_x = balanced_train.iloc[:,1:]
train_y = balanced_train.iloc[:,0]

test_x = balanced_test.iloc[:,1:]
test_y = balanced_test.iloc[:,0]





print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)





# Normalise data
x_train = np.array(train_x) / 255.0
y_train = np.array(train_y)
x_test = np.array(test_x) / 255.0
y_test = np.array(test_y)





x_train.astype('float32')
x_test.astype('float32')





#Reshaping all images into 28*28 for pre-processing
x_train = x_train.reshape(x_train.shape[0], 28, 28)
x_test = x_test.reshape(x_test.shape[0], 28, 28)
print(x_train.shape,x_test.shape)





def image(data):
        image = np.fliplr(data)
        image = np.rot90(image)
        return image

def create_images(data, start, end):
    images = []
    for i in range(start,end):
        images.append(image(data[i]))
    return images





y = create_images(x_train, 100,115)





rows = len(y)//2
cols = (len(y)//2)+1
axes=[]
fig=plt.figure(figsize=(16, 16))
for a in range(rows+cols):
    axes.append( fig.add_subplot(rows, cols, a+1))
    plt.imshow(y[a])
fig.tight_layout()    
plt.show()




import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras import utils
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import callbacks




# One hot encoding
train_y = to_categorical(train_y, classes)
test_y = to_categorical(test_y, classes)
print("train_y: ", train_y.shape)
print("test_y: ", test_y.shape)





# Reshape image for CNN
train_x = x_train.reshape(-1, 28, 28, 1)
test_x = x_test.reshape(-1, 28, 28, 1)
print("train_x_oo: ", train_x.shape)





# Partition to train and val
train_xx, label_x, train_yy, label_y = train_test_split(train_x, train_y,  test_size=0.2, random_state = 42)




model = Sequential()

model.add(layers.Conv2D(filters = 32,
                        kernel_size = (3,3),
                        padding = 'same',
                        activation = 'relu',
                        input_shape = (28, 28,1)))
model.add(layers.MaxPooling2D(pool_size = (2,2)))

model.add(layers.Conv2D(filters = 64,
                        kernel_size = (3,3),
                        padding = 'same',
                        activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size = (2,2)))
model.add(layers.Conv2D(filters = 64,
                        kernel_size = (3,3),
                        padding = 'same',
                        activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(units = 64,
                       activation = 'relu'))
model.add(layers.Dropout(.5))
model.add(layers.Dense(units = classes,
                       activation = 'softmax'))


# In[23]:


model.summary()


# In[24]:


model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# In[25]:


# Set callback functions to stop training early and save the best model so far
keras_callbacks   = [
    callbacks.EarlyStopping(monitor = 'val_loss', patience =50,
                            mode = 'auto', verbose = 1),
    callbacks.ModelCheckpoint(filepath = 'best_model_cnn_emnist.h5', monitor = 'val_loss',
                              save_best_only = True, mode = 'auto', verbose = 1)
]


# In[26]:


model_history = model.fit(train_xx, train_yy,
                    epochs = 100,
                    batch_size = 256,
                    verbose = 1,
                    validation_data=(label_x, label_y),
                         callbacks=keras_callbacks)


# In[27]:


# Variables - Function definition for plot accuracy
accuracy = model_history.history['accuracy']
val_accuracy = model_history.history['val_accuracy']
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
epochs = list(range(1,len(val_accuracy)+1))


# In[28]:


import json
data = {"accuracy":accuracy, "val_accuracy":val_accuracy, "loss":loss, "val_loss": val_loss, "epochs":epochs}
json_data = json.dumps(data)
with open("data.json", "w") as outfile:
    outfile.write(json_data)


# In[29]:


f = open("data.json", mode="r")
data = json.load(f)
accuracy = data["accuracy"]
val_accuracy = data["val_accuracy"]
loss = data["loss"]
val_loss = data["val_loss"]
epochs = data["epochs"]


# In[30]:


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=epochs,
    y=val_accuracy,
    name="Training"       # this sets its legend entry
))


fig.add_trace(go.Scatter(
    x=epochs,
    y=accuracy,
    name="Validation"
))

fig.update_layout(
    title="Model Accuracy",
    xaxis_title="Epoch",
    yaxis_title="Accuracy",
    legend_title="Accuracy vs Epochs",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)

fig.show()


# In[31]:


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=epochs,
    y=loss,
    name="Training"       # this sets its legend entry
))


fig.add_trace(go.Scatter(
    x=epochs,
    y=val_loss,
    name="Validation"
))

fig.update_layout(
    title="Model Loss",
    xaxis_title="Epoch",
    yaxis_title="Loss",
    legend_title="Loss vs Epochs",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)

fig.show()


# In[32]:


# Evaluate model
scores = model.evaluate(test_x, test_y, verbose=0)
print(f'Score: {model.metrics_names[0]} of {round(scores[0], 4)}; '
      f'{model.metrics_names[1]} of {round((scores[1]*100), 4)}%')


# In[33]:


y_pred = model.predict(test_x)


# In[34]:


# Confusion matrix (scikit-learn)
cm = metrics.confusion_matrix(test_y.argmax(axis=1), y_pred.argmax(axis=1))
cm


# In[35]:


from tensorflow.keras.models import load_model
saved_model = load_model("best_model_cnn_emnist.h5")


# In[36]:


# Evaluate the saved model
scores = saved_model.evaluate(test_x, test_y, verbose = 1)
print(f'Score: {saved_model.metrics_names[0]} of {round(scores[0], 4)}; '
      f'{saved_model.metrics_names[1]} of {round((scores[1]*100), 4)}%')

