
# coding: utf-8

# # Bike Demand Prediction

# In[1]:


# importing requried library

import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sn
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from math import sqrt
import time
import random
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
pd.options.mode.chained_assignment = None

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Reading dataset
raw_data = pd.read_csv("day.csv")


# In[3]:


raw_data.cnt.describe()


# In[4]:


raw_data.head(3)


# In[5]:


raw_data.shape


# We have 731 observations, 15 predictors and 1 target variable. Cnt is our target variable. 
# Next examining variable types

# In[6]:


raw_data.dtypes


# In the dataset season, yr, mnth, holiday, weekday, workingday, weathersit predictors should be categorical type, but they are int64. In the next step mapping and categorical transformatin will be performed.

# In[8]:


#converting to categorical variable

categorical_variable = ["season","yr","mnth","holiday","weekday","workingday","weathersit"]

for var in categorical_variable:
    raw_data[var] = raw_data[var].astype("category")


# In[9]:


raw_data.head(2)


# Droping variables which are not requried.
# 1. instant - index number
# 2. dteday- all the requried like month week day all ready present
# 3. casual and resgistered - their sum is equal to cnt ie. to the target variable

# In[10]:


raw_data = raw_data.drop(["instant","dteday"],axis = 1)


# ## Missing value Analysis

# In[11]:


# We will perform missing value andlysis using missingno package
msno.matrix(raw_data)


# There are no missing values present in the dataset

#    ## Outliers Analysis

# In[12]:


fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(10,12)
sn.boxplot(data=raw_data,y="cnt",orient='v',ax=axes[0][0])
sn.boxplot(data=raw_data,y="cnt",x="season",orient='v',ax=axes[0][1])
sn.boxplot(data=raw_data,y="cnt",x="weekday",orient="v",ax=axes[1][0])
sn.boxplot(data=raw_data,y="cnt",x="workingday",orient="v",ax=axes[1][1])

axes[0][0].set(ylabel='cnt',title = "Boxplot of cnt")
axes[0][1].set(xlabel="season",ylabel="cnt",title="Boxplot for cnt vs season")
axes[1][0].set(xlabel="weekday", ylabel="cnt",title="Boxplot for cnt vs weekday")
axes[1][1].set(xlabel="workingday",ylabel="cnt",title="Boxplot for cnt vs workingday")


# In[13]:


fig.set_size_inches(8,12)
sn.boxplot(data=raw_data, x="weathersit",y="cnt").set_title("Boxplot of cnt vs weathersit")


# From the above boxplots, it is evident that there are no outliers present in the cnt. Two things are clear.
# 1. Cnt is very low in spring season.
# 2. Cnt is maximum when weather is good and its minimum weather is bab.

# ## Correlation Analysis

# In[14]:


churn_corr = raw_data.corr()
cmap = cmap=sn.diverging_palette(15, 250, as_cmap=True)

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]

churn_corr.style.background_gradient(cmap, axis=1)    .set_properties(**{'max-width': '90px', 'font-size': '15pt'})    .set_caption("Correlation matrix")    .set_precision(2)    .set_table_styles(magnify())


# Finding of correlation analysis - 
# 1. temp and atemp are highly correlated.
# 2. temp and atemp have positive and strong coorelation with cnt.
# 3. hum and windspeed have negative and weak correlation with cnt.
# 

# ## Bivariate analysis

# In[15]:


# Bivariate analysis of cnt and continous predictor

fig,(ax1,ax2,ax3,ax4) = plt.subplots(ncols=4)
fig.set_size_inches(12,6)

sn.regplot(x="temp",y="cnt",data=raw_data,ax=ax1)
sn.regplot(x="atemp",y="cnt",data=raw_data,ax=ax2)
sn.regplot(x="hum",y="cnt",data=raw_data,ax=ax3)
sn.regplot(x="windspeed",y="cnt",data=raw_data,ax=ax4)


# From the above plot, it is evident that cnt has a positive linear relationship with temp and atemp.
# On the other hand, cnt has a negative linear relationship with windspeed.
# Humidity(hum) has a little negative linear relationship with cnt.

# ## Distribution of target Variable

# In[16]:


fig,(ax1,ax2) = plt.subplots(ncols=2)
fig.set_size_inches(9,5)
sn.distplot(raw_data["cnt"],ax=ax1)
stats.probplot(raw_data["cnt"], dist='norm', fit=True, plot=ax2)


# As we can see, out cnt variable is very close to normal distribution.

# Preprocessing original data and Spliting into train and test data

# In[17]:


# selecting predictors
train_feature_space = raw_data.iloc[:,raw_data.columns != 'cnt']
# selecting target class
target_class = raw_data.iloc[:,raw_data.columns == 'cnt']


# In[18]:


#droping atemp due to multicollinearity
#droping casual and registered because there sum is equal to target variable ie. 'cnt'

train_feature_space = train_feature_space.drop(["atemp","casual","registered"],axis = 1)


# In[19]:


train_feature_space.shape


# In[20]:


# creating training and test set
training_set, test_set, train_taget, test_target = train_test_split(train_feature_space,
                                                                    target_class,
                                                                    test_size = 0.30, 
                                                                    random_state = 456)

# Cleaning test sets to avoid future warning messages
train_taget = train_taget.values.ravel() 
test_target = test_target.values.ravel()


# ## Model1 Linear Regression Model
# 

# In[21]:


X = training_set
X = sm.add_constant(X) 
y= np.log(train_taget)

model = sm.OLS(y, X.astype(float)).fit()


# In[22]:


model.summary()


# In[23]:


# Initialize logistic regression model
lModel = LinearRegression()
lModel.fit(X = training_set,y = np.log(train_taget))


# In[24]:


#predicting using linear regression
lmPredictions = lModel.predict(X=test_set)


# In[25]:


x=pd.DataFrame(np.exp(lmPredictions))


# In[26]:


x.describe()


# In[27]:


lm_errors = abs(np.exp(lmPredictions) - test_target)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(lm_errors), 2), 'degrees.')


# In[28]:


rmse = sqrt(mean_squared_error(test_target, np.exp(lmPredictions)))


# In[29]:


print("RMSE for test set in linear regression is :" , rmse)


# In[30]:


## The line / model
plt.scatter(test_target, np.exp(lmPredictions))
plt.xlabel("True Values")
plt.ylabel("Predictions")


# ## Model2 Random forest

# In[31]:


rf = RandomForestRegressor(random_state=12345)


# In[32]:


rf


# In[33]:


np.random.seed(12)
start = time.time()

# selecting best max_depth, maximum features, split criterion and number of trees
param_dist = {'max_depth': [2,4,6,8,10],
              'bootstrap': [True, False],
              'max_features': ['auto', 'sqrt', 'log2',None],
              "n_estimators" : [100 ,200 ,300 ,400 ,500]
             }
cv_randomForest = RandomizedSearchCV(rf, cv = 10,
                     param_distributions = param_dist, 
                     n_iter = 10)

cv_randomForest.fit(training_set, train_taget)
print('Best Parameters using random search: \n', 
      cv_randomForest.best_params_)
end = time.time()
print('Time taken in random search: {0: .2f}'.format(end - start))


# In[34]:


# setting parameters

# Set best parameters given by random search # Set be 
rf.set_params( max_features = 'log2',
               max_depth =8 ,
               n_estimators = 300
                )


# In[35]:


rf.fit(training_set, train_taget)


# In[36]:


# Use the forest's predict method on the test data
rfPredictions = rf.predict(test_set)
# Calculate the absolute errors
rf_errors = abs(rfPredictions - test_target)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(rf_errors), 2), 'degrees.')


# In[37]:


rmse_rf = sqrt(mean_squared_error(test_target, rfPredictions))


# In[38]:


print("RMSE for test set in random forest regressor  is :" , rmse_rf)


# ### Variable importance for random forest

# In[39]:


feature_importance =  pd.Series(rf.feature_importances_, index=training_set.columns)
feature_importance.plot(kind='barh')


# In[41]:


#model input and output
pd.DataFrame(test_set).to_csv('InputLinearRegressionRandomForestPyhon.csv', index = False)
pd.DataFrame(np.exp(lmPredictions), columns=['predictions']).to_csv('outputLinearRegressionPython.csv')
pd.DataFrame(rfPredictions, columns=['predictions']).to_csv('outputRandomForestPython.csv')

