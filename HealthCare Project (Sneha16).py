#!/usr/bin/env python
# coding: utf-8

# In[181]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
import seaborn as sns


# In[182]:


data=pd.read_csv('health care diabetes.csv')


# In[183]:


data.head()


# In[184]:


data.shape


# ### Project Task: Week 1
# #### Data Exploration:
# 
# ##### Perform descriptive analysis. Understand the variables and their corresponding values. On the columns below, a value of zero does not make sense and thus indicates missing value:
# #### • Glucose
# 
# ##### • BloodPressure
# 
# ##### • SkinThickness
# 
# ##### • Insulin
# 
# ##### • BMI
# 
# ##### 1. Visually explore these variables using histograms. Treat the missing values accordingly.
# 
# ##### 2. There are integer and float data type variables in this dataset. Create a count (frequency) plot describing the data types and the count of variables.

# In[185]:


positive= data[data['Outcome']==1]
positive.head()


# In[186]:


data['Glucose'].value_counts().head()


# In[187]:


plt.hist(data['Glucose'])


# In[188]:


data['BloodPressure'].value_counts().head()


# In[189]:


plt.hist(data['BloodPressure'])


# In[190]:


data['SkinThickness'].value_counts().head()


# In[41]:


plt.hist(data['SkinThickness'])


# In[191]:


data['Insulin'].value_counts().head()


# In[192]:


plt.hist(data['Insulin'])


# In[193]:


data['BMI'].value_counts().head()


# In[45]:


plt.hist(data['BMI'])


# In[194]:


print('Number of missing values in Glucose :',data[data['Glucose']==0].shape[0])
print('Number of missing values in BloodPressure :',data[data['BloodPressure']==0].shape[0])
print('Number of missing values in SkinThickness :',data[data['SkinThickness']==0].shape[0])
print('Number of missing values in Insulin :',data[data['Insulin']==0].shape[0])
print('Number of missing values in BMI :',data[data['BMI']==0].shape[0])


# ### Observations :
# #### After analyzing the histogram we can identify that there are some outliers in some columns.
# #### For Example:-
# #### BloodPressure - A living person cannot have a diastolic blood pressure of zero.
# #### Plasma glucose levels - Zero is invalid number as fasting glucose level would never be as low as zero.
# #### Skin Fold Thickness - For normal people, skin fold thickness can’t be less than 10 mm better yet zero.
# #### BMI: Should not be 0 or close to zero unless the person is really underweight which could be life-threatening.
# #### Insulin: In a rare situation a person can have zero insulin but by observing
# ###### * We also found lots of 0. * **Number of missing values in (Glucose is 5), (BloodPressure is 35), (SkinThickness is 227), (Insulin is 374) & (BMI is 11).**

# ### Project Task: Week 2
# #### Data Exploration:
# 
# ##### 1.Check the balance of the data by plotting the count of outcomes by their value. Describe your findings and plan future course of action.
# 
# ##### 2.Create scatter charts between the pair of variables to understand the relationships. Describe your findings.
# 
# ##### 3.Perform correlation analysis. Visually explore it using a heat map.

# In[195]:


sns.countplot(data1.Outcome).set(title='Data balance check')


# ### Observation
# #### We can see this is a imbalanced dataset. Where the positive outcomes are half than the neagtive

# In[196]:


get_ipython().system(' pip install imbalanced-learn')


# In[199]:


from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']], df['Outcome'])


# In[200]:


df.columns


# In[201]:


pd.Series(y_resampled).value_counts().plot(kind='bar', title='Class distribution after appying SMOTE', xlabel='outcome')


# In[202]:


pd.Series(y_resampled).value_counts()


# In[203]:


X_resampled


# In[204]:


y_resampled


# In[205]:


X_resampled.isnull().sum().any()


# In[206]:


#Visualizing Pairplot
sns.pairplot(df,hue='Outcome')
plt.show()


# In[207]:


data1.corr()


# In[208]:


sns.heatmap(data1.corr(),annot=True)


# ### Correlation Analysis Observation:
# #### 1.The BMI-Skinthickness and Insulin-Glucose are the highest correlated in the set but they are moderately correlated
# #### 2.Outcome is moderately correlated to Glucose
# #### 3.Age is moderately correlated to pregnancies

# ### Project Task: Week 3
# #### Data Modeling:
# 
# ##### 1. Devise strategies for model building. It is important to decide the right validation framework. Express your thought process.
# 
# ##### 2. Apply an appropriate classification algorithm to build a model. Compare various models with the results from KNN algorithm.

# In[209]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[210]:


# define the features and outcome
X = data.drop('Outcome', axis=1)
y = data['Outcome']


# In[211]:


# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)


# In[212]:


#Logistic Regression


# In[213]:


# initialize the models
logreg = LogisticRegression()

# fit the models to the training data
logreg.fit(X_train, y_train)

# make predictions on the test set
y_pred_logreg = logreg.predict(X_test)

# calculate the accuracy of the models
acc_logreg = accuracy_score(y_test, y_pred_logreg)

# print the accuracy of the models
print("Logistic Regression accuracy :}",acc_logreg)


# In[214]:


logreg.score(X_test,y_test)


# In[215]:


from sklearn.svm import SVC


# In[216]:


# initialize the models
svm = SVC()

# fit the models to the training data
svm.fit(X_train, y_train)

# make predictions on the test set
y_pred_svm = svm.predict(X_test)

# calculate the accuracy of the models
acc_svm = accuracy_score(y_test, y_pred_svm)

# print the accuracy of the models
print("SVM accuracy :",acc_svm)


# In[217]:


from sklearn.tree import DecisionTreeClassifier


# In[218]:


# initialize the models
dt = DecisionTreeClassifier()

# fit the models to the training data
dt.fit(X_train, y_train)

# make predictions on the test set
y_pred_dt = dt.predict(X_test)

# calculate the accuracy of the models
acc_dt = accuracy_score(y_test, y_pred_dt)

# print the accuracy of the models
print("Decision Tree accuracy :",acc_dt)


# In[219]:


from sklearn.ensemble import RandomForestClassifier


# In[220]:


# initialize the models
rf = RandomForestClassifier()

# fit the models to the training data
rf.fit(X_train, y_train)

# make predictions on the test set
y_pred_rf = rf.predict(X_test)

# calculate the accuracy of the models
acc_rf = accuracy_score(y_test, y_pred_rf)

# print the accuracy of the models
print("Random Forest accuracy :",acc_rf)


# In[160]:


from sklearn.neighbors import KNeighborsClassifier

# create the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)

# fit the classifier to the training data
knn.fit(X_train, y_train)

# predict on the test set
y_pred = knn.predict(X_test)

# evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("KNeighborsClassifier accuracy:", accuracy)


# In[221]:


data.to_csv('health.csv')


# ### Project Task: Week 4
# #### Data Modeling:
# #### 1. Create a classification report by analyzing sensitivity, specificity, AUC (ROC curve), etc. Please be descriptive to explain what values of these parameter you have used.

# In[222]:


from sklearn.metrics import classification_report, roc_auc_score, roc_curve


# In[223]:


# Classification report by analyzing sensitivity, specificity, AUC (ROC curve) for LogisticRegression


# In[224]:


# generate the classification report
report = classification_report(y_test, y_pred_logreg)
print(report)


# In[225]:


# calculate AUC curve
auc = roc_auc_score(y_test, y_pred_logreg)
print('AUC: %.3f' % auc)

# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_logreg)
print("True Positive Rate:",tpr,"\nFalse Positive Rate:",fpr,"\nThresholds:",thresholds)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# In[226]:


# Classification report by analyzing sensitivity, specificity, AUC (ROC curve) for SVC


# In[227]:


# generate the classification report
report = classification_report(y_test, y_pred_svm)
print(report)


# In[228]:


# calculate AUC curve
auc = roc_auc_score(y_test, y_pred_svm)
print('AUC: %.3f' % auc)

# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_svm)
print("True Positive Rate:",tpr,"\nFalse Positive Rate:",fpr,"\nThresholds:",thresholds)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# In[229]:


# Classification report by analyzing sensitivity, specificity, AUC (ROC curve) for Decision Tree


# In[230]:


# generate the classification report
report = classification_report(y_test, y_pred_dt)
print(report)


# In[231]:


# calculate AUC curve
auc = roc_auc_score(y_test, y_pred_dt)
print('AUC: %.3f' % auc)

# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_dt)
print("True Positive Rate:",tpr,"\nFalse Positive Rate:",fpr,"\nThresholds:",thresholds)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# In[232]:


# Classification report by analyzing sensitivity, specificity, AUC (ROC curve) for Random Forest Classifier


# In[233]:


# generate the classification report
report = classification_report(y_test, y_pred_rf)
print(report)


# In[234]:


# calculate AUC curve
auc = roc_auc_score(y_test, y_pred_rf)
print('AUC: %.3f' % auc)

# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_rf)
print("True Positive Rate:",tpr,"\nFalse Positive Rate:",fpr,"\nThresholds:",thresholds)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# In[235]:


# Classification report by analyzing sensitivity, specificity, AUC (ROC curve) for KNeighborsClassifier


# In[236]:


# generate the classification report
report = classification_report(y_test, y_pred)
print(report)


# In[237]:


# calculate AUC curve
auc = roc_auc_score(y_test, y_pred)
print('AUC: %.3f' % auc)

# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_logreg)
print("True Positive Rate:",tpr,"\nFalse Positive Rate:",fpr,"\nThresholds:",thresholds)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# In[ ]:




