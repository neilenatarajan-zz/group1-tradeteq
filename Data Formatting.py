#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pickle
import pandas as pd
import numpy as np


# In[2]:


filename = "co_T0_T1_toshare.pkl"
area_to_minority = "OAtoMinorityData.CSV"


# In[3]:


pickle_in = open(filename,"rb")
df1 = pickle.load(pickle_in)
df2 = pd.read_csv(area_to_minority)


# In[14]:


df = pd.merge(df1,df2)


# In[15]:


from IPython.display import display
pd.options.display.max_columns = None
df.head()


# In[16]:


#keep only those companies whose T0 status is active
df = df[df['CompanyStatus_T0']=='Active']


# In[24]:


features = df[['CompanyNumber', 'oac2',
        'CountryOfOrigin','MortgagesNumMortCharges',
       'MortgagesNumMortOutstanding', 'MortgagesNumMortPartSatisfied', 'LimitedPartnershipsNumGenPartners',
       'LimitedPartnershipsNumLimPartners',  'SIC1',
       'eCompanyCategory', 'eCompanyStatus',
       'eAccountsAccountCategory', 'dIncorporationDate',
       'dAccountsNextDueDate',
       'dReturnsNextDueDate','dConfStmtNextDueDate','namechanged', 'namechanged2', 'class_1','MinorityScore']].copy()
#Remove the rows with missing data 
features.dropna(how='any',inplace=True)


# In[25]:


#Create new features(periods)
T0 = 2018+(31+28)/365
T1 = 2019+(31+28)/365
features['dAccountsOutstandingTime'] = T0-features['dAccountsNextDueDate']
features['dConfStmtOutstandingTime'] = T0-features['dConfStmtNextDueDate']
features['dReturnsOutstandingTime'] = T0-features['dReturnsNextDueDate']
features['OperatingTime'] = T1-features['dIncorporationDate']
features.loc[features['dAccountsOutstandingTime']<0, 'dAccountsOutstandingTime'] = 0 
features.loc[features['dConfStmtOutstandingTime']<0, 'dConfStmtOutstandingTime'] = 0 
features.loc[features['dReturnsOutstandingTime']<0, 'dReturnsOutstandingTime'] = 0
features.drop(['dAccountsNextDueDate','dConfStmtNextDueDate','dReturnsNextDueDate','dIncorporationDate'],axis = 1,inplace = True)
features.head()


# In[27]:


features['dConfStmtOutstandingTime'].value_counts()


# In[28]:


features['CountryOfOrigin'].value_counts()


# In[29]:


features['LimitedPartnershipsNumGenPartners'].value_counts()


# In[30]:


features['eCompanyStatus'].value_counts()


# In[31]:


features['LimitedPartnershipsNumLimPartners'].value_counts()


# In[32]:


features['dAccountsOutstandingTime'].value_counts()


# In[33]:


features.drop(['dConfStmtOutstandingTime','CountryOfOrigin','LimitedPartnershipsNumGenPartners','LimitedPartnershipsNumLimPartners','eCompanyStatus'],axis = 1,inplace = True)


# In[34]:


features = pd.get_dummies(features, columns=['SIC1', 'oac2','eCompanyCategory','eAccountsAccountCategory'])


# In[35]:


features.head()


# In[37]:


pickle.dump(features, open("processed_data.pkl", "wb" ) )


# In[ ]:




