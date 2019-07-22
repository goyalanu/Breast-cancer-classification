#Importing Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer

cancer.keys()#keys of dataset

print(cancer['DESCR'])#discription of dataset

print(cancer['target'])#target of dataset (0 and 1 for target names)
 
print(cancer['target_names'])#target names :- malignant and benign

print(cancer['feature_names'])#feature_names:- radius,texture,perimeter,area etc

cancer['data'].shape#actual data i.e shape :-(569,30)

df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'],['target']))#dataframe for dataset
df_cancer.head()#it gives first 5 enteries
df_cancer.tail()#it gives last 5 enteries

#visualizzing the data

sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])
sns.countplot(df_cancer['target'])
sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)
plt.figure(figsize = (20, 10))
sns.heatmap(df_cancer.corr(), annot = True)

#model training
x = df_cancer.drop(['target'], axis = 1)
y = df_cancer['target']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
svc_model = SVC()
svc_model.fit(x_train, y_train)

#Evaluating the model
y_predict = svc_model.predict(x_test)
cmat = confusion_matrix(y_test, y_predict)
sns.heatmap(cmat, annot = True)

#Improving the model
min_train = x_train.min()
range_train = (x_train-min_train).max()
x_train_scaled = (x_train - min_train)/range_train
sns.scatterplot(X = x_train['mean area'], y = x_train['mean smoothness'], hue = y_train)
sns.scatterplot(X = x_train_scaled['mean area'], y = x_train_scaled['mean smoothness'], hue = y_train)
min_test = x_test.min()
range_test = (x_test - min_test).max()
x_test_scaled = (x_test - min_test)/range_test
svc_model.fit(x_train_scaled, y_train)
y_predict = svc_model.predict(x_test_scaled)
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot = True)
print(classification_report(y_test, y_predict))
