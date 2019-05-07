#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This line will make sure that results are replicable

from numpy.random import seed
seed(42)
#from tensorflow import set_random_seed
#set_random_seed(42)


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf


print("libraries all imported, ready to go")


# In[3]:


filename = 'processed_data.pkl'
feature = 'class_1'


# In[4]:


pickle_in = open(filename,"rb")
df = pickle.load(pickle_in)


# In[5]:


df = df.drop(['CompanyNumber'], axis=1)


# In[6]:


df.head()


# In[7]:


y = df[feature].values
y.shape


# In[8]:


X = df.drop(feature, axis=1).values
X.shape


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[10]:


X_test


# In[11]:


X_test_minority = X_test[np.where(X_test[:,5] >= 0.9)]


# In[12]:


y_test_minority = y_test[np.where(X_test[:,5] >= 0.9)]


# In[13]:


X_test_minority = np.delete(X_test_minority, 5, 1)
X_train = np.delete(X_train, 5, 1)
X_test = np.delete(X_test, 5, 1)


# In[14]:


X_test.shape


# In[15]:


X_test_minority.shape


# In[16]:


X_test_minority


# In[17]:


y_test_minority


# In[18]:


# Describe the architecture of the model, except for the input layer.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(20, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
])


# set some additional settings
model.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[19]:


history = model.fit(X_train, 
                    y_train, 
                    epochs=2, 
                    batch_size=512,
                    verbose=0)


# In[20]:


loss, acc = model.evaluate(X_test, y_test)
print('Loss: {:0.3f}, Accuracy: {:0.3f}'.format(loss, acc) )


# In[21]:


loss, acc = model.evaluate(X_test_minority, y_test_minority)
print('Loss: {:0.3f}, Accuracy: {:0.3f}'.format(loss, acc) )


# In[22]:


y_pred = model.predict(X_test)


# In[23]:


y_pred_minority = model.predict(X_test_minority)


# In[24]:


fpr1, tpr1, _ = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.roc_auc_score(y_test, y_pred)


# In[25]:


plt.figure()
plt.plot(fpr1, tpr1, color='red', label='ROC curve (area = %0.6f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[26]:


fpr2, tpr2, _ = metrics.roc_curve(y_test_minority, y_pred_minority)
roc_auc = metrics.roc_auc_score(y_test_minority, y_pred_minority)


# In[27]:


plt.figure()
plt.plot(fpr2, tpr2, color='red', label='ROC curve (area = %0.6f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




