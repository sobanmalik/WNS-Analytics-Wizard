#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from warnings import filterwarnings
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_absolute_error,accuracy_score,roc_auc_score
from sklearn.neural_network import MLPRegressor
train,test=pd.read_csv('train.csv'),pd.read_csv('test.csv')
view_log,item_data=pd.read_csv('view_log.csv'),pd.read_csv('item_data.csv')
view_log=view_log.drop(['server_time','session_id'],axis=1)


# ## Merging and Cleaning Data

# In[3]:


train.shape,test.shape,view_log.shape,item_data.shape


# In[4]:


train.head()


# In[5]:


view_log.head()


# In[6]:


item_data.head()


# In[7]:


def merge(df):
    df=pd.merge(df,view_log,on='user_id',how='left')
    df=pd.merge(df,item_data,on='item_id',how='left')
    df=df.dropna(subset=['impression_id'])
    return df


# In[8]:


tic=time.time()
train_1=merge(train).drop_duplicates().sort_values('impression_id').reset_index(drop=True)
test_1=merge(test).drop_duplicates().sort_values('impression_id').reset_index(drop=True)
toc=time.time()
print('time:{} sec'.format(toc-tic))


# In[9]:


def impute_mean(df,columns):
    for i in columns:
        df[i].fillna(df[i].mean(),inplace=True)
def impute_median(df,columns):
        df[columns].fillna(df[columns].median(),inplace=True)
columns=train_1.columns[-4:]
impute_mean(train_1,columns)
impute_mean(test_1,columns)
columns=train_1.columns[-5]
impute_median(train_1,columns)
impute_median(test_1,columns)


# In[10]:


train_1.shape,train.shape


# In[11]:


train_1.describe().round(1)


# ## Removing Outliers

# In[12]:


plt.boxplot(train_1['item_price'])
#plt.boxplot(test_1['item_price'].dropna())
plt.show()


# In[13]:


from scipy import stats
z = np.abs(stats.zscore(train_1['item_price']))
train_1=train_1.loc[z<3,:]


# In[14]:


train_1['item_price'].plot.hist()


# ## Grouping Data

# In[15]:


def group(df):
    columns=train_1.columns[8:]
    a=df.groupby(['impression_id'],as_index=False)[columns].mean()
    b=df.drop(columns,axis=1).drop_duplicates(subset=['impression_id'],keep='last')
    d_f=pd.merge(b,a,on='impression_id', how='left')
    return d_f


# In[16]:


train_1


# In[17]:


train_final=group(train_1).round(1)
test_final=group(test_1).round(1)


#    ## Plotting Features
#     

# In[18]:


plt.subplot(plt.)
sns.heatmap(train_X.corr(),cmap='Blues')


# In[19]:


click=train_final[train_final['is_click']==1]
click['user_id'].plot.hist(alpha=0.9,color='black',bins=100)
click['item_id'].plot.hist(alpha=0.7,color='yellow',bins=100)
click['item_price'].plot.hist(alpha=0.7,color='blue',bins=100)
plt.show()


# Hence should drop user_id because not much disparity

# In[20]:


train_final=train_final.drop(columns='user_id')
test_final=test_final.drop(columns='user_id')


# In[21]:


click['product_type'].plot.hist(alpha=0.9,color='red',bins=100)
click['app_code'].plot.hist(alpha=0.7,color='blue',bins=100)
plt.show()


# Should not drop

# In[ ]:


#PIVOTING
#train_final.pivot_table(index='os_version',values='is_click').plot.bar()
#train_final.pivot_table(index='device_type',values='is_click').plot.bar()
#plt.show()


# ## Feature Engineering

# In[22]:


train_final.head()


# In[23]:


test_final.describe().round(1)


# In[24]:


train_final['item_price'].plot.hist()
#test_final['product_type'].head(1000).plot.hist()


# In[25]:


def price_cut(df,cut_points,labels):
    df['price_categories']=pd.cut(df['item_price'],cut_points,labels=labels)
    return df
cut_points=[0,2000,6000,8500,12500,20000,35000,45000,301901]
labels=['cheap4','cheap3','cheap2','cheap1','expensive1','expensive2','expensive3','expensive4']
train_final=price_cut(train_final,cut_points,labels)
test_final=price_cut(test_final,cut_points,labels)


# In[26]:


train_final.describe().round(1)


# In[27]:


from datetime import datetime as dt
t=list()
t.append(dt.strptime(train_final['impression_time'][i], '%Y-%m-%d %H:%M:%S').strftime('%m,%d') for i in range(train_final.shape[0]))
train_final['month_day']=[dt.strptime(train_final['impression_time'][i], '%Y-%m-%d %H:%M:%S').strftime('%m,%d') for i in range(train_final.shape[0])]
train_final['hour']=[int(dt.strptime(train_final['impression_time'][i], '%Y-%m-%d %H:%M:%S').strftime('%H')) for i in range(train_final.shape[0])]
test_final['month_day']=[dt.strptime(test_final['impression_time'][i], '%Y-%m-%d %H:%M:%S').strftime('%m,%d') for i in range(test_final.shape[0])]
test_final['hour']=[int(dt.strptime(test_final['impression_time'][i], '%Y-%m-%d %H:%M:%S').strftime('%H')) for i in range(test_final.shape[0])]


# In[ ]:


train_final.head()


# In[28]:


def time_cut(df,cut_points,labels):
    df['time_categories']=pd.cut(df['hour'],cut_points,labels=labels)
    return df
cut_points=[-1,5,12,17,21,24]
labels=['late_night','morning','afternoon','evening','night']
train_final=time_cut(train_final,cut_points,labels)
test_final=time_cut(test_final,cut_points,labels)


# In[29]:


def get_day(month,day):
    #day_list=[]
    if month==11:
        dates=[i for i in range(15,22)]
        days=['Thursday','Friday','Saturday','Sunday','Monday','Tuesday','Wednesday']
        for i in range(len(dates)):
            if day==dates[i] or day==dates[i]+7  or day==dates[i]+14:
                #day_list.append(days[i])
                day_final=days[i]
    if month==12:
        dates=[i for i in range(1,8)]
        days=['Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday','Friday']
        for i in range(len(dates)):
            if day==dates[i] or day==dates[i]+7  or day==dates[i]+14 or day==dates[i]+21:
                #day_list.append(days[i])
                day_final=days[i]
    return day_final
    


# In[30]:


train_final['day']=[get_day(int(train_final.month_day[i].split(',')[0]),int(train_final.month_day[i].split(',')[1])) for i in range(len(train_final.index))]
test_final['day']=[get_day(int(test_final.month_day[i].split(',')[0]),int(test_final.month_day[i].split(',')[1])) for i in range(len(test_final.index))]


# In[ ]:


train_final.head()


# In[31]:


#PIVOTING
train_final.pivot_table(index='time_categories',values='is_click').plot.bar()
train_final.pivot_table(index='day',values='is_click').plot.bar()
#plt.show()


# In[32]:


def create_dummies(df,column_name):
    dummy_table=pd.get_dummies(df[column_name],prefix='Class')
    df=pd.concat([df,dummy_table],axis=1)
    return df
for i in ['time_categories','day','price_categories','os_version','device_type']:
    train_final=create_dummies(train_final,i)
    test_final=create_dummies(test_final,i)


# In[33]:


from sklearn.preprocessing import minmax_scale
for i in ['item_id','category_1','category_2','category_3','product_type']:
    train_final[i+'_scaled']=minmax_scale(train_final[i])
    test_final[i+'_scaled']=minmax_scale(test_final[i])
train_final['is_4g']=train_final['is_4G']
test_final['is_4g']=test_final['is_4G']


# In[35]:


import seaborn as sns
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=50)
sns.heatmap(train_final[list(train_final.iloc[:5,18:48].columns)+['is_click']].astype(float).corr().round(1),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# ## Modelling

# In[43]:


get_ipython().run_line_magic('time', '')
columns=train_final.iloc[:,18:].columns
train_X,test_X,train_y,test_y=train_test_split(train_final[columns]
                            ,train_final['is_click'],test_size=0.2,random_state=42)


# In[52]:


get_ipython().run_line_magic('time', '')
from catboost import CatBoostRegressor
cb_model = CatBoostRegressor(iterations=500,
                             learning_rate=0.002,
                             depth=12,
                             eval_metric='AUC',
                             random_seed = 23,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = 75,
                             od_wait=100)
cb_model.fit(train_X, train_y,
             eval_set=(test_X,test_y),
             #cat_features=categorical_var,
             use_best_model=True,
             verbose=True)


# In[41]:


#convert into binary values
for i in range(y_pred.shape[0]):
    if y_pred[i]>=.5:       # setting threshold to .5
        y_pred[i]=1
    else:  
        y_pred[i]=0


# In[48]:


accuracy_score(y_pred,train_y)


# In[101]:


from sklearn.model_selection import StratifiedKFold
def Stacking(model,train,y,test,n_fold):
    folds=StratifiedKFold(n_splits=n_fold,random_state=1)
    test_pred=np.empty((test.shape[0],1),float)
    train_pred=np.empty((0,1),float)
    for train_indices,val_indices in folds.split(train,y.values):
        x_train,x_val=train.iloc[train_indices],train.iloc[val_indices]
        y_train,y_val=y.iloc[train_indices],y.iloc[val_indices]

    model.fit(X=x_train,y=y_train)
    train_pred=np.append(train_pred,model.predict(x_train))
    test_pred=np.append(test_pred,model.predict(test))
    return test_pred.reshape(-1,1),train_pred


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
model2 = SGDClassifier(loss='hinge', penalty="l2", max_iter=100)
test_pred2 ,train_pred2=Stacking(model=model2,n_fold=10, train=train_X,
                                 test=test_X,y=train_y)
train_pred2=pd.DataFrame(train_pred2)
test_pred2=pd.DataFrame(test_pred2)


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
model3 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.001,
                 max_depth=1, random_state=0).fit(train_X, train_y)
test_pred3 ,train_pred3=Stacking(model=model3,n_fold=10, train=train_X,
                                 test=test_X,y=train_y)
train_pred3=pd.DataFrame(train_pred3)
test_pred3=pd.DataFrame(test_pred3)


# In[86]:


model4 =  RandomForestClassifier(n_estimators=10, max_depth=None,max_features="sqrt",
           min_samples_split=2, random_state=0)
test_pred4 ,train_pred4=Stacking(model=model4,n_fold=10, train=train_X,
                                 test=test_X,y=train_y)
train_pred4=pd.DataFrame(train_pred4)
test_pred4=pd.DataFrame(test_pred4)


# In[94]:


df_test.shape,test_y.shape


# In[63]:


model1 = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
model1.fit(train_X, train_y)
val_pred1=model1.predict(test_X)
test_pred1=model1.predict(test_final[columns])
val_pred1=pd.DataFrame(val_pred1)
test_pred1=pd.DataFrame(test_pred1)

model2 = SGDClassifier(loss='hinge', penalty="l2", max_iter=100)
model2.fit(train_X,train_y)
val_pred2=model2.predict(test_X)
test_pred2=model2.predict(test_final[columns])
val_pred2=pd.DataFrame(val_pred2)
test_pred2=pd.DataFrame(test_pred2)


# In[ ]:


model3 =  GradientBoostingClassifier(n_estimators=100, learning_rate=0.001,
                 max_depth=1, random_state=0).fit(train_X, train_y)
model3.fit(train_X,train_y)
val_pred3=model3.predict(test_X)
test_pred3=model3.predict(test_final[columns])
val_pred3=pd.DataFrame(val_pred3)
test_pred3=pd.DataFrame(test_pred3)

model4 = RandomForestClassifier(n_estimators=10, max_depth=None,max_features="sqrt",
           min_samples_split=2, random_state=0)
model4.fit(train_X,train_y)
val_pred4=model4.predict(test_X)
test_pred4=model4.predict(test_final[columns])
val_pred4=pd.DataFrame(val_pred4)[0].dtype('int')
test_pred4=pd.DataFrame(test_pred4)


# In[ ]:


df_val=pd.concat([test_X,val_pred1,val_pred2,val_pred4,val_pred4],axis=1)
df_test=pd.concat([ test_final,test_pred1,test_pred2,test_pred3,test_pred4],axis=1)

model = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
model.fit(df_val,test_y)
#model.score(df_test,y_test)


# In[ ]:


df = pd.concat([train_pred1, train_pred2, train_pred3, train_pred4], axis=1)
df_test = pd.concat([test_pred1, test_pred2, test_pred3, test_pred4], axis=1)
model = LogisticRegression(random_state=42)
model.fit(df,train_y[:-3])
#model.score(df_test, y_test)


# In[133]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
clf= DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
scores = cross_val_score(clf, train_X, train_y, cv=5)
scores.mean()   


# In[160]:


clf.fit(train_X,train_y)
print('test:' + str(clf.score(test_X,test_y).round(4)),
      'train:' + str(clf.score(train_X,train_y).round(4)))


# In[161]:


clf = RandomForestClassifier(n_estimators=10, max_depth=None,max_features="sqrt",
    min_samples_split=2, random_state=0)
scores = cross_val_score(clf, train_X, train_y, cv=5)
scores.mean()                               


# In[175]:


from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.001,
    max_depth=1, random_state=0).fit(train_X, train_y)
clf.score(test_X, test_y)                 


# In[61]:


predictions=model.predict(test_X)
accuracy=accuracy_score(predictions,test_y)
accuracy


# In[56]:


scores=cross_val_score(model,train_final[columns],train_final['is_click'],cv=5,scoring='roc_auc')
accuracy_cv=scores.mean()
accuracy_cv


# In[60]:


clf.coef_


# In[62]:


feature_importance=pd.DataFrame(clf.coef_[0],index=columns)
feature_importance.plot.barh()


# In[63]:


feature_importance_sorted=feature_importance.abs().sort_values(0,ascending=False)
feature_importance_sorted.plot.barh()


# In[104]:


score_list=list()
for i in range(1,9):
    columns_imp=feature_importance_sorted.index[:i]
    clf = SGDClassifier(loss="log", penalty="l2", max_iter=10)
    clf.fit(train_X[columns_imp],train_y)
    scores=cross_val_score(clf,train_final[columns_imp],train_final['is_click'],cv=10)
    accuracy_cv=scores.mean()
    score_list.append(accuracy_cv)
score_df=pd.DataFrame(score_list,index=range(1,9))
score_df


# In[164]:


tic=time.time()
scores=cross_val_score(clf,train_final[columns],train_final['is_click'],cv=10)
accuracy_cv=scores.mean()
print(accuracy_cv)
toc=time.time()
print('time:{}'.format(toc-tic) + ' sec')


# In[141]:


test_final['item_price'].plot.hist()


# In[57]:


holdout_predictions=model.predict(test_final[columns])
holdout_predictions


# In[58]:


submission_dict={'impression_id':test_final['impression_id'],
              'is_click':holdout_predictions}
submission_df=pd.DataFrame(data=submission_dict)
submission_df.to_csv('Submission1.csv',index=False)


# In[59]:


submission_df.is_click.unique()

