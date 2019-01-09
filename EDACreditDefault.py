#!/usr/bin/env python
# coding: utf-8

# In[14]:


##Exploratory Data Analysis
##Probability of Customer Credit Default Data Set 

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns 


# In[13]:


#Import Data
credit = pd.read_csv('defaultcc.csv', header = 0)
credit1 = pd.read_csv('defaultccNUM.csv', header = 0)
cc1 = credit1

ccORD = pd.read_csv('OrdinalValuesDefaultCC.csv')


# In[5]:


credit.head()


# In[8]:


#examine data
credit.describe()


# In[152]:


#identify values 
credit.info()


# In[6]:


cc = pd.read_csv("defaultcc.csv", header = 0)


# In[10]:


cc.head()


# In[5]:


cc1.info()


# In[4]:


import matplotlib.pyplot as plt


# In[7]:


#SHOW COLUMN NAMES#

header = cc.dtypes.index
print(header)


# In[11]:


#BUILD HISTOGRAM#

plt.hist(cc['LIMIT_BAL'])
plt.show()

#Shows the value of Limit Balance throughout the entire data set


# In[160]:


#Histogram with 4 bins 

plt.hist(cc['LIMIT_BAL'], bins = 4)
plt.show()


# In[161]:


#LINE PLOT GRAPH#

plt.plot(cc['LIMIT_BAL'])
print()


# In[12]:


##Scatter PLot##

x = ccORD['PAST_PAY_1']
y = ccORD['PAST_PAY_2']

plt.scatter(x,y)
plt.show()


# In[10]:


# Scatte rplot Age and Education

a = cc['AGE']
e = cc['EDUCATION']

plt.scatter(e, a)
plt.show()


# In[163]:


#Print Headers

header = cc.dtypes.index
print(header)


# In[7]:


#BOXPLOT

A = cc['AGE']
plt.boxplot(A, 0, 'gD')
plt.show()


# In[12]:


#BOXPLOT Past_Pay1
A = cc['PAST_PAY1']
plt.boxplot(A, 0, 'gD')
plt.show()


# In[11]:


#ccORD boxplot pastpay1
B = ccORD['PAST_PAY_1']
plt.boxplot(B, 0, 'gD')
plt.show()


# In[8]:


#CORRELATION MATRIX

corrMat = cc.corr()
print(corrMat)


# In[19]:


#Export corrMat
corrMat.to_csv('corrMat1.csv')


# In[8]:


#Visulaize and remove high coreelated features

corrMatNUM = cc1.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corrMatNUM,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(corrMatNUM.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(corrMatNUM.columns)
ax.set_yticklabels(corrMatNUM.columns)
plt.show()



# In[52]:


#COVARIANCE 

covMat = cc1.cov()
print(covMat)

#export
covMat.to_csv('covMat.csv')


# In[15]:


#Visulaize and remove high coreelated features

covMatNUM = ccORD.cov()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(covMatNUM,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(covMatNUM.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(covMatNUM.columns)
ax.set_yticklabels(covMatNUM.columns)
plt.show()


# In[16]:


#Comparing attribute values

pastpay1 = ccORD.groupby('SEPT_STATUS')['SEPT_STATUS'].count()
pastpay1


# In[17]:


#Past_Pay_1 ccORD
pastpay1ccORD = ccORD.groupby('PAST_PAY_1')['PAST_PAY_1'].count()
pastpay1ccORD


# In[13]:


#compare education
education = cc.groupby('EDUCATION')['EDUCATION'].count()
education


# In[15]:


#education ccORD
educationccORD = ccORD.groupby('EDUCATION')['EDUCATION'].count()
educationccORD


# In[77]:


#Comparing attribute values

cc.groupby('SEX')['SEX'].count()


# In[166]:


#Comparing attribute values
cc.groupby('EDUCATION')['EDUCATION'].count()


# In[79]:


#Comparing attribute values
cc.groupby('MARRIAGE')['MARRIAGE'].count()


# In[80]:


#Plot the education levels 

fg = sns.factorplot('EDUCATION', data = cc, kind = 'count', aspect = 1.5)
fg.set_xlabels('EDUCATION')


# In[55]:


#Plot the education levels 
#updated code with 'catplot' instead of 'factorplot'

fg = sns.catplot('EDUCATION', data = cc, kind = 'count', aspect = 1.5)
fg.set_xlabels('EDUCATION')


# In[82]:


fg1 = sns.catplot('MARRIAGE', data = cc, kind = 'count', aspect = 1.5)
fg1.set_xlabels('MARRIAGE')


# In[168]:


#Use seaborn to group by Marriage Status and Education

fg2 = sns.catplot('EDUCATION', data = cc, hue = 'MARRIAGE', kind = 'count', aspect = 1.75)
fg2.set_xlabels('EDUCATION')


# In[99]:


#Use seaborn to group by Marriage Status and Age
fg3 = sns.catplot('AGE', data = cc, hue = 'MARRIAGE', kind = 'count', aspect = 1.75)


# In[103]:


#number of customers who defaulted by education level and marriage status 
cc.pivot_table('DEFAULTED', 'EDUCATION', 'MARRIAGE', aggfunc=np.sum, margins = True)


# In[9]:


#create an object for customer who did not default
not_default = cc[cc['DEFAULTED']==0]
default = cc[cc['DEFAULTED']==1]


# In[10]:


#Factor plot of those who defaulted and those who did not default

sns.catplot('DEFAULTED', data = cc, kind = 'count')


# In[11]:


#total number of customers who did not default
len(not_default)


# In[12]:


len(default)


# In[121]:


#customers who defaulted and who didn't default grouped by marriage status and education
table = pd.crosstab(index=[cc.DEFAULTED,cc.EDUCATION], columns=[cc.SEX, cc.MARRIAGE])


# In[122]:


table.unstack()


# In[116]:


table.columns, table.index


# In[126]:


# Change name of columns
table.columns.set_levels(['Female', 'Male'], level=0, inplace=True)
table.columns.set_levels(['Other', 'Married', 'Single', 'Divorce'], level=1, inplace=True)
table


# In[128]:


print('Average and median age of passengers are %0.f and %0.f years old, respectively'%(cc.AGE.mean(), 
                                                                           cc.AGE.median()))


# In[129]:


#describe AGE

cc.AGE.describe()


# In[130]:


#drop missing values for the records in which age passenger is missing
age = cc['AGE'].dropna()


# In[131]:


#Distribution of age, with an overlay of a density plot
age = cc['AGE'].dropna()
age_dist = sns.distplot(age)
age_dist.set_title("Distribution of Customers Ages")


# In[132]:


#Another way to plot histogram of age
cc['AGE'].hist(bins=50)


# In[23]:


#create a function to identify male/female
def male_female(passenger):
    SEX = passenger
    
    if SEX > 1.5:
        return female
    else:
        return male

##-OR-##


# In[39]:


#Change the names of the values in certain attributes, within the attribute column in excel (SEX, EDUCATION, MARRIAGE)

cc['SEX'] = cc['SEX'].astype(object)
pd.factorize(cc.SEX)
cc['SEX'] = 


# In[19]:


#Value Counts
cc['SEX'].value_counts()


# In[146]:


cc[:5]


# In[169]:


#do a factorplt of customers sex, marriage status, and education
sns.catplot('EDUCATION', data = cc, kind = 'count', hue = 'SEX')


# In[43]:


#kde plot, Distribution of Passengers' Age
#Grouped by Gender

fig = sns.FacetGrid(cc, hue = 'SEX', aspect = 4)
fig.map(sns.kdeplot, 'AGE', shade = True)
oldest = cc['AGE'].max()
fig.set(xlim=(0,oldest))
fig.set(title='Distribution of Age Grouped by Gender')
fig.add_legend()


# In[54]:


#kde plot, Distibution of Passenger Education

fig1 = sns.FacetGrid(cc, hue = 'EDUCATION', aspect = 4)
fig1.map(sns.kdeplot, 'AGE', shade = True)
oldest1 = cc['AGE'].max()
fig1.set(xlim=(0, oldest1))
fig1.set(title='Distribution of Age Grouped by Education')
fig1.add_legend()


# In[72]:


#kdeplot, Distribution of Credit Limit by Education 

fig2 = sns.FacetGrid(cc, hue = 'EDUCATION', aspect = 4)
fig2.map(sns.kdeplot, 'LIMIT_BAL', shade = True)
oldest2 = cc['LIMIT_BAL'].max()
fig2.set(xlim=(0, oldest2))
fig2.set(title='Distribution of Credit Limit Grouped by Education')
fig2.add_legend()


# In[53]:


#kdeplot, Distribution of Credit Limit by MArriage Status 

fig3 = sns.FacetGrid(cc, hue = 'MARRIAGE', aspect = 4)
fig3.map(sns.kdeplot, 'LIMIT_BAL', shade = True)
oldest3 = cc['LIMIT_BAL'].max()
fig3.set(xlim=(0, oldest3))
fig3.set(title='Distribution of Credit Limit Grouped by Marriage Status')
fig3.add_legend()


# In[83]:


## Defaulted vs. Marriage Status grouped by Education

#change Marriage back to number to plot
credit1 = pd.read_csv('defaultccMARRIAGEnum.csv', header = 0)

sns.catplot(x = 'MARRIAGE', y ='DEFAULTED', hue = 'EDUCATION', data = credit1, order = range(1,5),
            hue_order = ['high school', 'university', 'grad school', 'other'])


# In[95]:


#Discretize Credit Limit into 4 bins of 25% percentiles

#set up bins
bin = [10000, 50000, 140000, 240000, 1000000]
#use pd.cut function can attribute the values into specific bins
category = pd.cut(cc.LIMIT_BAL, bin)
category = category.to_frame()

#concatenate age and its bin 
cc_new = pd.concat([cc, category], axis = 1)


#ERROR, look into how to execute pd.cut


# In[44]:


#Create Age bins

generations = [30, 40, 50, 60]
sns.lmplot('AGE', 'DEFAULTED', hue='EDUCATION', data=cc, x_bins=generations)


# In[38]:


print(cc_new)


# In[89]:


cc_new.head()


# In[7]:


#linear plot of age vs Defaulted

sns.lmplot('AGE', 'DEFAULTED', data=cc, hue = 'SEX')


# In[40]:


#Guassian curve manufactured for Age 

# histogram plot of a low res sample
from numpy.random import seed
from numpy.random import randn
from matplotlib import pyplot
from numpy import exp
from scipy.stats import boxcox
# seed the random number generator
seed(1)

#define data
data1 = cc.AGE
#power transform
data = boxcox(data1, 0)

#transform to be expoential 
#data = exp(data1)

pyplot.hist(data)
pyplot.show()


# In[5]:


##RFE - Recursive Feature Elimination

from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from sklearn import datasets

svm = LinearSVC()
# create the RFE model for the svm classifer and select attributes
rfe = RFE(svm, 24)
rfe = rfe.fit(ccORD, ccORD.DEFAULTED)

# print summaries for the selection of attributes
print(rfe.support_)
print(rfe.ranking_)


# In[18]:


#Plotting univariate Distirbution
sns.distplot(cc.LIMIT_BAL)


# In[34]:


#linear plot of age vs Defaulted

sns.lmplot('LIMIT_BAL', 'DEFAULTED', data=cc, hue = 'SEX')


# In[20]:


#linear plot of age vs Defaulted

sns.lmplot('AGE', 'DEFAULTED', data=cc, hue = 'EDUCATION')


# In[47]:


#linear plot of age vs Defaulted

sns.lmplot('AGE', 'DEFAULTED', data=cc, hue = 'MARRIAGE')


# In[23]:


sns.lmplot('PAST_PAY1', 'DEFAULTED', data=cc, hue = 'EDUCATION')


# In[50]:


sns.lmplot('PAST_PAY1', 'DEFAULTED', data=cc, hue = 'MARRIAGE')


# In[49]:


sns.lmplot('PAST_PAY2', 'DEFAULTED', data=cc, hue = 'SEX')


# In[25]:


sns.lmplot('PAST_PAY3', 'DEFAULTED', data=cc, hue = 'EDUCATION')


# In[31]:


sns.lmplot('PAST_PAY2', 'DEFAULTED', data=cc, hue = 'MARRIAGE')


# In[ ]:




