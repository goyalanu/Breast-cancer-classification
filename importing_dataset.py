Importing Dataset
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
