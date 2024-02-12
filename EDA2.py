# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 08:52:07 2023

@author: Admin

Business Objectives=Domain: Real Estate

Challenges:

Missing value treatment
Outlier treatment
Understanding which variables drive the price of homes in Boston


The Boston housing dataset contains 506 observations and 14 variables.
The dataset contains missing values.

Maximum:Finding the maximum profit
Minimum:
Business constraint:


"""

import pandas as pd
import seaborn as sns
import numpy as np
#Importing the csv file of boston dataset
df=pd.read_csv("C:/Datasets/Boston.csv.xls")
#Finding shape of the data set i.e how many row and columns are present in that dataset
df.shape
#To descibe the structure of the column
df.describe
#Finding the dataset of the column
df.dtypes
#Finding the columns of the dataset
df.columns
#Finding the duplicate are there in the dataset or not 
duplicate=df.duplicated()
#Printing the duplicates
duplicate
#If their is the duplicates values then we are taking 
#Then we are taking the sum of the duplicate values 
#using the following processs
sum(duplicate)
#
df.head
#Finding the missing  values in the given datasets
df.isnull
#Fro finding the first qurant

q1=df.crim.quantile(0.25)
q1
q3=df.crim.quantile(0.75)
q3
#For finding the iqr
iqr=q3-q1
iqr
sns.boxplot(df.crim)
lower_limit=df.crim.quantile(0.25)-1.5*iqr
upper_limit=df.crim.quantile(0.75)+1.5*iqr
out=np.where(df.crim>upper_limit,True,np.where(df.crim<lower_limit,True,False))
df_trimmed=df.loc[~out]
df.shape

df_trimmed.shape

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['crim'])
#Copy the winsorizer and paste in the help tab 
#top of the right window studdy the method
df_t=winsor.fit_transform(df[['crim']])
sns.boxplot(df['crim'])
sns.boxplot(df_t['crim'])






######################################################



