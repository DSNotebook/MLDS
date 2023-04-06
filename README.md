#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pickle
import flask
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from io import StringIO
from IPython.display import Image
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from   sklearn.ensemble        import RandomForestClassifier 
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score,confusion_matrix,\
                            classification_report,roc_auc_score,roc_curve,precision_score
from sklearn.datasets import load_iris

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.preprocessing import MinMaxScaler,LabelEncoder


file=r"winequalityN.csv"

df=pd.read_csv(file)
df.fillna(df.mean(),inplace=True)

enc=LabelEncoder()
df["type"]=enc.fit_transform(df['type'])
df

X=df.iloc[:,:-1]
y=df.quality
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=10)

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor


minmax_scaler=MinMaxScaler()
minmax_scaler.fit(X_train)
X_train=minmax_scaler.transform(X_train)

X_test=minmax_scaler.transform(X_test)



#model=DecisionTreeClassifier(criterion='gini',max_depth=4)
model=RandomForestClassifier(n_estimators=25, verbose=2, random_state = 10)
#model=LinearRegression()
model.fit(X_train,y_train)

ypred=model.predict(X_train)


pickle.dump(model,open("wineQualitypredict.pkl",'wb'))
pickle.dump(minmax_scaler,open("wineQuality_input.pkl",'wb'))

Adv_model=pickle.load(open("wineQualitypredict.pkl",'rb'))
input_scaler=pickle.load(open("wineQuality_input.pkl",'rb'))


pd.crosstab(y_train,ypred)

accuracy_score(y_train,ypred)

print(classification_report(y_train,ypred))


feat=model.feature_importances_
feat
l=list(feat)
l=pd.DataFrame(l,columns=['list'])
name=X.columns
plt.figure(figsize=(15,5))
plt.grid()
l['name']=list(X)
#l.plot(kind='barh')
sns.swarmplot(x=l.name,y=l.list,data=l,)
sns.scatterplot(x=l.name,y=l.list,data=l,hue=l.name)#df.type


#l=[]
#for i in range(X.shape[1]):
 #   fe=input(X.columns[i])
#    l.append(fe)

#winequality=model.predict([l])
#if(winequality>=7):
#    print(winequality[0],'is','Quality is good')
#else:
#    print(winequality[0],'is','bad quality')
    


# In[5]:


bestfea=set(l.list)
type(bestfea)
bestfea=list(bestfea)
bestfea=max(bestfea[:-1])#in top best #1 feature
featurename=l[l.list==bestfea].iloc[:,1:]
featurename=str(featurename)
len(featurename[:-7])
featurename=featurename[16:]
featurename[:]

feat=model.feature_importances_
feat.max()
feat=model.feature_importances_
feat
l=list(feat)

l=pd.DataFrame(l,columns=['list'])
l['name']=list(X)
bestfea=set(l.list)
featurename=l[l.list==feat.max()]
featurename


# In[11]:


set(l.list)


# In[14]:


df.plot()


# import matplotlib.pyplot as plt   # To plot a graph
# from matplotlib import rcParams
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import scipy as sc

# sns.displot(df.TV)

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import scale
# from sklearn.model_selection import cross_val_score
# from collections import Counter
# from sklearn.preprocessing import MinMaxScaler,LabelEncoder
# 

# from sklearn.neural_network import MLPRegressor
# from sklearn.ensemble import RandomForestRegressor
# 

# import pickle
# import flask

# #data=pd.read_csv("Advertising.csv")
# 
# file=r"C:\Users\ravik\Data Files\1. ST Academy - Crash course and Regression files\House_Price.csv"
# data=pd.read_csv(file)

# # Split the data

# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)#random_state=0

# # scale the data

# minmax_scaler=MinMaxScaler()
# minmax_scaler.fit(X_train)
# X_train=minmax_scaler.transform(X_train)
# 
# X_test=minmax_scaler.transform(X_test)

# In[3]:


model_LR=LinearRegression()
#model_LR=MLPRegressor(hidden_layer_sizes=(60,50,20))
#model_LR=RandomForestRegressor(n_estimators=100,random_state=10)


# In[4]:


model_LR.fit(X_train,y_train)


# In[5]:


predict=model_LR.predict(X_train)


# # dump the model

# pickle.dump(model_LR,open("Adverting_model_pickle.pkl",'wb'))

# pickle.dump(minmax_scaler,open("Adverting_input_scaler.pkl",'wb'))

# # load the model 

# Adv_model=pickle.load(open("Adverting_model_pickle.pkl",'rb'))

# input_scaler=pickle.load(open("Adverting_input_scaler.pkl",'rb'))

# Directory structure 
# Projectory Name /
# templates/
# index.html/
# app.py/
# module.py

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




