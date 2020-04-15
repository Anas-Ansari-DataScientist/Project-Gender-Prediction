#!/usr/bin/env python
# coding: utf-8

# # Data Importing

# In[91]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# In[92]:


data=pd.read_csv('weight-height.csv')
data


# In[93]:


X=data.iloc[:,[1,2]].values
y=data.iloc[:,0].values


# In[94]:


from scipy import stats
z = np.abs(stats.zscore(X))
print(z)
threshold = 3
print(np.where(z > 3))
new_data = data[(z < 3).all(axis=1)]
new_data


# # Data Visualization  

# In[95]:


#Seperating Male and Female Data
data1=new_data[new_data['Gender']=='Male']
data2=new_data[new_data['Gender']=='Female']


# In[116]:


data1


# In[97]:


data1.shape


# In[98]:


#data1  # for male 
data2.shape  #for female


# In[99]:


#Data is balanced or not ?
sns.set_style('whitegrid')
sns.countplot(x='Gender',data=new_data,palette='RdBu_r')


# In[100]:


new_data.hist()


# In[101]:


data1=new_data[new_data['Gender']=='Male']
data2=new_data[new_data['Gender']=='Female']
plt.scatter(data1['Height'],data1['Weight'],color='K',s=25,marker='*') #All the data with Y=1,represented by Black,Admitted
plt.scatter(data2['Height'],data2['Weight'],color='R',s=25,marker='o') # All data with Y=0 ,represented by Red,Rejected
plt.title('Visualization of Data',fontsize=12,style='italic',fontweight='bold')
plt.xlabel('Height',fontsize=12)
plt.ylabel('Weight',fontsize=12)
plt.show()


# # Data Processing

# In[143]:


height=X["Height"]/12
weight=X['Weight']*0.453592


# In[161]:


df = pd.DataFrame(height) 
df2 = pd.DataFrame(weight) 
X = pd.concat(df,df2)


# In[165]:


frames=[df,df2]
X = pd.concat(frames,axis=1)


# In[166]:


X.values
y=new_data.iloc[:,0].values


# In[103]:


from sklearn.preprocessing import LabelEncoder  # Encoding library call
enc = LabelEncoder()
label_encoder = enc.fit(y)
label_encoder
Y = label_encoder.transform(y)
Y


# In[167]:


## Spliting your data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)


# In[135]:


X_train


# # Model Building

# In[106]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = 14)
    score=cross_val_score(knn_classifier,X_train,Y_train,cv=10)
    knn_scores.append(score.mean())
score.mean()


# In[107]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty = 'l2', solver='sag',multi_class='multinomial',class_weight='balanced',random_state=42)  #solver=lbfgh is gradient decent
clf.fit(X_train,Y_train)


# In[108]:


from sklearn.model_selection import cross_val_score
cross_val_score(clf, X_test, Y_test, cv=10, scoring="accuracy").mean()


# In[168]:


from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=250,criterion = 'entropy',max_depth=7, min_samples_split = 7)
rnd_clf.fit(X_train, Y_train)
y_pred_rf = rnd_clf.predict(X_test)


# In[112]:


from sklearn.model_selection import cross_val_score
cross_val_score(rnd_clf, X_test, Y_test, cv=10, scoring="accuracy").mean()


# In[113]:


from sklearn.tree import DecisionTreeClassifier
DecisionTreeClassifier= DecisionTreeClassifier(max_depth=4,criterion = 'gini',min_samples_leaf = 1,min_samples_split= 2, max_leaf_nodes = 10)

score=cross_val_score(DecisionTreeClassifier,X_test,Y_test,cv=10)
score.mean()


# In[174]:


X_train


# In[175]:


Y_train


# In[169]:


# Saving model to disk
pickle.dump(rnd_clf, open('model.pkl','wb'))


# In[177]:


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[5.75,73.93]]))

