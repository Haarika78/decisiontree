#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv("diabetes.csv")


# In[2]:


df.head()


# In[3]:


df.shape


# In[4]:


df.isnull().sum()


# In[12]:


X=df.iloc[:,:-1].to_numpy()
y=df.iloc[:,-1].to_numpy()


# In[13]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[33]:


from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)


# In[34]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.tree import plot_tree
plt.figure(figsize=(20,10))
plot_tree(clf,feature_names=['Glucose','BMI'],class_names=['No','Yes'])
plt.show()


# In[36]:


clf.set_params(max_depth=3)


# In[37]:


clf.fit(X_train,y_train)
plt.figure(figsize=(20,10))
plot_tree(clf,feature_names=['Glucose','BMI'],class_names=['No','Yes'])
plt.show()


# In[38]:


predictions=clf.predict(X_test)


# In[39]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(clf,X_train,y_train,cv=5,scoring='accuracy')
accuracy=scores.mean()
accuracy


# In[40]:


from sklearn import metrics
cf=metrics.confusion_matrix(y_test,predictions)
cf


# In[41]:


tp=cf[1][1]
tn=cf[0][0]
fp=cf[0][1]
fn=cf[1][0]
print(f"tp:{tp},tn:{tn},fp:{fp},fn:{fn}")


# In[42]:


print("accuracy",metrics.accuracy_score(y_test,predictions))


# In[43]:


print("precision",metrics.precision_score(y_test,predictions))


# In[44]:


print("Recall",metrics.recall_score(y_test,predictions))


# In[45]:


feature_importances=clf.feature_importances_
print("Feature importances:",feature_importances)


# In[ ]:




