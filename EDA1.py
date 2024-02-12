# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 09:32:46 2023

@author: Admin
"""

import pandas as pd
df=pd.read_csv("C:/Datasets/ethnic diversity.csv")
#Let us check the data types of the column
df.dtypes
#Salaraies data type is float let us convert into int
df.Salaries=df.Salaries.astype(int)
df.dtypes
#Now the dara type of salaries is int similarly age data type must be float presently it is int
df.age=df.age.astype(float)
df.dtypes



################################################
#Identifty the duplicates
df_new=pd.read_csv("C:/Datasets/education.csv.xls")
duplicate=df_new.duplicated()

#output of this function is single column
#if there is duplicate records output-True
#if there is duplicste records output-False
duplicate
sum(duplicate)




#############################################
new=pd.read_csv("C:/Datasets/mtcars_dup.csv.xls")
duplicate=new.duplicated()
duplicate
sum(duplicate)
#THere are 3 duplicate reecords
#Row 
df1=new.drop_duplicates()
df2=df1.duplicated()
df2
sum(df2)


################################################
#Outlier treatment
import pandas as pd
import seaborn as sns
df=pd.read_csv("C:/Datasets/ethnic diversity.csv")
#Now let us find the outlier in the salaries
sns.boxplot(df.Salaries)
#There are no outliers
#Let us check outliers in age column
sns.boxplot(df.age)
#There are no outliers
#Let us calculte IQR
IQR=df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)
#Let us calculate IQR




lower_limit=df.Salaries.quantile(0.25)-1.5*IQR
upper_limit=df.Salaries.quantile(0.75)+1.5*IQR
#now if you will check the upper limit

















############################################################
#Trimming
#Remoiving all the outliers

import numpy as np
out=np.where(df.Salaries>upper_limit,True,np.where(df.Salaries<lower_limit,True,False))
#If our column is greater than upper limit then return True 
#If our column is less than lower limit then return True
#Else return False
#You can check the outliers column in variable explorer
df_trimmed=df.loc[~out]
df.shape

df_trimmed.shape

#Drawback of trimming techinique is we are losing the data

####################################################
#Replacment Technique

df=pd.read_csv("C:/Datasets/ethnic diversity.csv")
df.describe()


#Record no 23 has got outliers
# map all the outliers values to upper limit
df_replaced=pd.DataFrame(np.where(df.Salaries>upper_limit,upper_limit ,np.where(df.Salaries<lower_limit,lower_limit,df.Salaries)))
#If the values are greater than upper limit
#map it to the upper limit and less than lower limit
#map it to the lower limit,if it is witnin the range
#THen keep as it is
sns.boxplot(df_replaced[0])


#####################################################
#Winsorizer
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Salaries'])
#Copy the winsorizer and paste in the help tab 
#top of the right window studdy the method
df_t=winsor.fit_transform(df[['Salaries']])
sns.boxplot(df['Salaries'])
sns.boxplot(df_t['Salaries'])
