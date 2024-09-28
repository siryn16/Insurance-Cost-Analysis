#!/usr/bin/env python
# coding: utf-8

# # Practice Project : Insurance Cost Analysis

# In this project, you have to perform analytics operations on an insurance database that uses the below mentioned parameters.
# 
# | Parameter | Description | Content type |
# |-----------|-------------|--------------|
# | age            | Age in years  | integer |
# | gender         | Male or Female  | integer (1 or 2)  |
# | bmi            | Body mass index | float  |
# | no_of_children | Number of children | integer  |
# | smoker  | Wether smoker or not | integer (0 or 1)  |
# | region | Which US region - NW, NE, SW, SE | integer (1,2,3 or 4 respectively)  |
# | charges | Annual insurance charges in USD  | float  |

# ### Objectives

# In this project, you will:
# 
# * Load the data as a pandas dataframe
# * Clean the data, taking care of the blank entries
# * Run exploratory data analysis (EDA) and identify the attributes that most affect the charges
# * Develop single variable and multi variable Linear Regression models for predicting the charges
# * Use Ridge regression to refine the performance of Linear regression models.

# ### Setup

# For this lab, we will be using the following libraries:
# 
# * `skillsnetwork` to download the data
# * `pandas` for managing the data.
# * `numpy` for mathematical operations.
# * `sklearn` for machine learning and machine-learning-pipeline related functions.
# * `seaborn` for visualizing the data.
# * `matplotlib` for additional plotting tools.

# #### Importing required libraries

# In[3]:


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split


# #### Download the dataset to this lab environment

# In[13]:


filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv'
df = pd.read_csv(filepath, header=None)


# ### Task 1 : Import the Dataset

# Import the dataset into a `pandas` dataframe. Note that there are currently no headers in the CSV file.
# 
# Print the first 10 rows of the dataframe to confirm successful loading.

# In[14]:


df.head(10)


# Add the headers to the dataframe, as mentioned in the project scenario.

# In[15]:


df.columns = ["age", "gender", "bmi", "no_of_children", "smoker", "region", "charges"]


# In[23]:


df.head(10)


# Now, replace the '?' entries with 'NaN' values.

# In[16]:


df.isna().sum()


# In[17]:


(df == '?').sum()


# In[18]:


df.replace('?', np.nan, inplace=True)


# In[19]:


df.isna().sum()


# ### Task 2: Data Wrangling

# In[20]:


df.info()


# Handle missing data:
# 
# * For continuous attributes (e.g., age), replace missing values with the mean.
# * For categorical attributes (e.g., smoker), replace missing values with the most frequent value.
# * Update the data types of the respective columns.
# * Verify the update using `df.info()`.

# In[25]:


#Continious Attribute
mean_age = df['age'].astype(float).mean(axis=0)
df['age'].replace(np.nan, mean_age, inplace=True)


# In[22]:


#Categorical Value
is_smoker = df['smoker'].value_counts().idxmax()
df['smoker'].replace(np.nan, is_smoker, inplace=True)


# In[29]:


#updating the datatypes
df[['age', 'smoker']]= df[['age', 'smoker']].astype(int)


# In[30]:


df.info()


# Also note, that the ``charges`` column has values which are more than 2 decimal places long. Update the `charges` column such that all values are rounded to nearest 2 decimal places. Verify conversion by printing the first 5 values of the updated dataframe.

# In[32]:


df[['charges']] = np.round(df[['charges']],2)
df.head(10)


# ### Task 3 : Exploratory Data Analysis (EDA)

# Implement the regression plot for ``charges`` with respect to ``bmi``.

# In[33]:


sns.regplot(x='bmi', y='charges', data=df, line_kws ={'color' : 'red'})
plt.ylim(0,)


# Implement the box plot for ``charges`` with respect to ``smoker``.

# In[37]:


sns.boxplot(x ='smoker', y='charges', data=df)


# Print the correlation matrix for the dataset.

# In[38]:


df.corr()


# ### Task 4 : Model Developement

# Fit a linear regression model that may be used to predict the `charges` value, just by using the `smoker` attribute of the dataset. Print the R^2
#  score of this model.

# In[39]:


X = df[['smoker']]
Y = df[['charges']]
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X,Y)


# Fit a linear regression model that may be used to predict the ``charges`` value, just by using all other attributes of the dataset. Print the R^2
# score of this model. You should see an improvement in the performance.

# In[41]:


Z = df[['age', 'gender', 'bmi', 'no_of_children', 'smoker', 'region']]
lm.fit(Z,Y)
lm.score(Z,Y)


# Create a training pipeline that uses ``StandardScaler()``, ``PolynomialFeatures()`` and ``LinearRegression()`` to create a model that can predict the charges value using all the other attributes of the dataset. There should be even further improvement in the performance.

# In[47]:


Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]
pipe = Pipeline(Input)
Z = Z.astype(float)
pipe.fit(Z,Y)
ypipe = pipe.predict(Z)
r2_score(Y, ypipe) #use r2_score with pipelines : r2_score(y_true, y_pred)


# ### Task 5 : Model Refinement

# Split the data into training and testing subsets, assuming that 20% of the data will be reserved for testing.

# In[48]:


x_train, x_test, y_train, y_test = train_test_split(Z, Y, test_size =0.2, random_state =1)


# Initialize a Ridge regressor that used hyperparameter alpha = 0.1. Fit the model using training data data subset. Print the 
# R^2 score for the testing data.

# In[49]:


RidgeModel = Ridge(alpha=0.1)
RidgeModel.fit(x_train, y_train)
yhat = RidgeModel.predict(x_test)
r2_score(y_test, yhat)


# Apply polynomial transformation to the training parameters with degree=2. Use this transformed feature set to fit the same regression model, as above, using the training subset. Print the R^2 score for the testing subset.

# In[53]:


pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform (x_train)
x_test_pr = pr.transform (x_test)
RidgeModel.fit(x_train_pr, y_train)
y_hat = RidgeModel.predict(x_test_pr)
r2_score(y_test, y_hat)

