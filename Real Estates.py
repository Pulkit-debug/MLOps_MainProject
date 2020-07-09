#!/usr/bin/env python
# coding: utf-8

# # Real Estate - Price Predictor

# In[1]:


import pandas as pd


# In[3]:


housing = pd.read_csv("houseData.csv")


# In[4]:


housing.head()


# In[5]:


housing.info()


# In[6]:


housing['CHAS'].value_counts()


# In[7]:


housing.describe() 
# count : This ignores the null values and show the others with some values
# mean : average 
# std : standard deviation
# here 25%, 50% and 75% means that the values that is shows are less than 25%, 50% and 75%


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


# # For plotting histogram
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20, 15))


# ## Train-Test Splitting

# In[10]:


from sklearn.model_selection import train_test_split
train_set, test_set  = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[11]:


#all done but here is something else that as we saw in our CHAS feature we had 0 - 471 and 1 - 35 so if we split our train and test
# this way there is a possibility that all the zeros go to our test and this could land in bad or wrong predictions So here we use a
# concept of Statified Sampling


# In[12]:


#So here we are now using StratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[13]:


strat_test_set['CHAS'].value_counts()
#So now here as we can see now it's taking both 0 and 1 values


# In[14]:


strat_train_set['CHAS'].value_counts()


# In[15]:


test_check = 95/7
train_check = 376/28
print("test ratio: " + str(test_check) + " and " + "test ratio: " + str(train_check))
#Here we can see almost same ratio we have of test and train
# same classification ratio is very import  while doing regression


# In[16]:


# before looking for corrletions it's good to take a copy of your dataset
housing = strat_train_set.copy()


# ## Looking for Correlations

# In[17]:


#pearson correlation coefficient
corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)
# 1 means strong positive correlation and as we can see MEDV(actual price) is 1
# 0.69 we can also see in RM(number of rooms per dwelling)(maximum the number of rooms maximum the price) 
#so we can see how well our corrleation is matching with our data
#pearson corrleation lies between -1 to 1
#LSTAT(% of lower status population) is -0.73 which shows that less the number of lower status people the more will be the pricing in the area which makes sense.


# In[22]:


# from pandas.plotting import scatter_matrix
from pandas.plotting import scatter_matrix

attributes = ["MEDV", "RM", "ZN", "LSTAT"]
scatter_matrix(housing[attributes], figsize = (12,8))

#here we are using only these four because we can't plot everyone because 14*14 makes a total of 196 combinations


# In[23]:


#In the above scatter we can clearly see that how LSTAT is decreasing and MEDV is increasing at LSTAT decreases


# In[24]:


housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8)

#here alpha is for density


# In[25]:


#Now here comes a thought if we somehow able to remove these outlets that are scatterd in the graph away from the increasing pattern
# we can give us a clean dataset and our machine learning will not be confused because of these points and will learn good pattern
#And the prediciton will be also good
#we have a capping at 50 
# we can see the points at 50 betweeen MEDV and RM those points are not good because if the rooms then the price must also increase
#which is not happening there


# ## Trying out Attribute combinations
# 

# In[26]:


# here we are creating a new features i.e TAX per room (TAXRM)
housing["TAXRM"] = housing['TAX']/housing['RM']


# In[27]:


#we can enhance our dataset using attribute combinations


# In[28]:


housing.head()


# In[29]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[30]:


# here we can see we have got a very good feature TAXRM is second after LSTAT
housing.plot(kind="scatter", x="TAXRM", y = "MEDV", alpha=0.8)
# we can see we have got a very strong negative relation between our TAXRM and MEDV


# In[31]:


# NOTE: In our Dataset we don't have any missing values so there is no need to worry aoutr missing attributes but we had some 
  #  missing attributes then we can use "Imputer"  imputer is very helpful in these case 


# In[32]:


housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()


# In[33]:


#Now we are creating a pipleine which will help automate our program a bit 
#means we are writing the code so that we can make changes easily later on our code, change strategies etc.


# ## Creating a Pipeline

# In[34]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #     ..... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])


# In[35]:


housing_num_tr = my_pipeline.fit_transform(housing)


# In[36]:


housing_num_tr.shape


# ## Selecting a desired model for Dragon Real Estates

# In[37]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)


# In[38]:


some_data = housing.iloc[:5]


# In[39]:


some_labels = housing_labels.iloc[:5]


# In[40]:


prepared_data = my_pipeline.transform(some_data)


# In[41]:


model.predict(prepared_data)


# In[42]:


list(some_labels)


# ## Evaluating the model

# In[43]:


from sklearn.metrics import mean_squared_error
import numpy as np
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)


# In[44]:


rmse


# ## Using better evaluation technique - Cross Validation

# In[45]:


# 1 2 3 4 5 6 7 8 9 10
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)


# In[46]:


rmse_scores


# In[47]:


def print_scores(scores):
    print("Scores:", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())


# In[48]:


print_scores(rmse_scores)


# ## Saving the model

# In[49]:


from joblib import dump, load
dump(model, 'realEstate.joblib') 


# ## Testing the model on test data

# In[55]:


X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_predictions, list(Y_test))


# In[56]:


final_rmse
# here we are getting our root mean squraerd error approx 3 which is very very good as compared to the data that we took
# we took only 500 data so our model is more than good here.


# In[57]:


prepared_data[0]


# ## Using the model

# In[58]:


from joblib import dump, load
import numpy as np
model = load('realEstate.joblib') 
features = np.array([[-5.43942006, 4.12628155, -1.6165014, -0.67288841, -1.42262747,
       -11.44443979304, -49.31238772,  7.61111401, -26.0016879 , -0.5778192 ,
       -0.97491834,  0.41164221, -66.86091034]])
model.predict(features)
#Here this is the price that we are getting for the features that we gave


# In[ ]:




