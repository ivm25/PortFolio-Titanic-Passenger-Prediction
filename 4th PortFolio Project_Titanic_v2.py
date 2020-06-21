#!/usr/bin/env python
# coding: utf-8
# File imported as a Jupyter File and working in spyder for testing and will work on this version
# ### Table of Contents
# ***
# 1 [DEFINE](#definition)
# 
# 1.1 [BUSINESS PROBLEM](#problem)
# 
# 2 [DISCOVER](#discover)
# 
# 2.1 [Loading](#loadthefile)
# 
# 2.2 [Exploring the data](#etl)
# 
# 2.3 [Visualisations using seaborn, plotly](#Visualise)
# 
# 2.4 [Feature Engineering](#Engineer)
# 
# 2.5 [Preprocessing dataset for model creation](#preprocess)
# 
# 2.5.1 [Creation of dummy variables](#dummies)
# 
# 2.5.2 [Dealing With Outliers (IQR, Zscore)](#outliers)
# 
# 3 [DEVELOP](#BaselineModel)
# 
# 3.1 [ModelDevelopment](#development)
# 
# 3.2 [ModelEvaluation](#evaluate)
# 
# 3.3 [ModelTuning](#tuning)
# 
# 4 [DEPLOY](#deploy)
# 

# ### 1: DEFINE
# <a id="definition"></a>

# **Classifying Titanic Passengers into two categories (Survived or not survived), by creatig a classification model that learns from the historical Shipwreck data. In addition, we will identify the correlation of the survival index with the different variables like Age, fare etc and deliver actionable insights from the existing dataset.**
# ***
# 
# *STATING THE ASSUMPTIONS:*
#     
# 
# 
# *The data is a digital collection of the most discussed shipwreck, i.e. Titanic and it is assumed that the entries are correct for analysis purposes.*
# 

# ### 1.1 BUSINESS PROBLEM
# <a id="problem"></a>

# **Classify Passengers of the test set into survived or not survived in this classic ML dataset.**
# ***
# 

# *The train dataset includes three 12 categories or variables.*
# 
# The columns in this dataset are:
# 
# 1. Passenger: IdUnique ID of the passenger
# 
# 2. Survived: Survived (1) or died (0)
# 
# 3. Pclass: Passenger's class (1st, 2nd, or 3rd)
# 
# 4. Name: Passenger's name
# 
# 5. Sex: Passenger's sex
# 
# 6. Age: Passenger's age
# 
# 7. SibSp: Number of siblings/spouses aboard the Titanic
# 
# 8. Parch: Number of parents/children aboard the Titanic
# 
# 9. Ticket: Ticket number
# 
# 10. Fare: Fare paid for ticket
# 
# 11. Cabin: Cabin number
# 
# 12. Embarked: Where the passenger got on the ship (C - Cherbourg, S - Southampton, Q = Queenstown)

# ### 2: DISCOVER
# <a id="discover"></a>

# *LAYING DOWN THE GROUNDWORK*
# ***
# - What are we analyzing?
# 
# **In the first step, we will be analyzing the distributions of the different variables of the dataset and write down the inferences. We will look for missing values and impute them.**
# 
# - What our variables mean?
# 
# **The different variables signify the characteristics of all the passengers that boarded the Titanic.**
# 
# - Why are we analyzing this data set?
# 
# **To develop a working classification model that can be deployed and scaled up. In addition, the predictions will help us estimate the accuracy of the baseline models.**

# ### 2.1 Loading the data
# <a id="loadthefile"></a>

# In[1]:


#import your libraries
import pandas as pd
import sklearn as sk
import numpy as np
import scipy
from scipy import stats
import plotly.express as px
import sqlite3
import re
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


# In[3]:


#Connecting to the database server
conn=sqlite3.connect('C:\\Data Science\\DSDJ\\Fourth PortFolio Project_Titanic_Kaggle\\Titanic.db')

#Creating an object of the connection
cur=conn.cursor()

#Running a query to extract all the data from the databse and saving it as a dataframe
Titanic_Train_table = pd.read_sql_query("SELECT * FROM Train", conn)

#Running a query to extract all the data from the databse and saving it as a dataframe
Titanic_Test_table = pd.read_sql_query("SELECT * FROM Test", conn)

#Querying the train table to identify blank cells
Titanic_Train_Blanks=pd.read_sql_query('SELECT * FROM Train WHERE AGE LIKE "" OR EMBARKED LIKE "" OR NAME LIKE "" OR CABIN LIKE "" OR SURVIVED LIKE "" OR Pclass LIKE "" OR SEX LIKE "" ', conn)

#Querying the test table to identify blank cells
Titanic_Test_Blanks=pd.read_sql_query('SELECT * FROM Test WHERE AGE LIKE "" OR EMBARKED LIKE "" OR NAME LIKE "" OR CABIN LIKE "" OR Pclass LIKE "" OR SEX LIKE "" ', conn)

#Inspect the Blank cell Train dataframe
Titanic_Train_Blanks

#Inspect the Blank cell Test dataframe
Titanic_Test_Blanks


# In[4]:


class Load:
    def __init__(self, df):
        """Initialising the constructor to save data in the different attributes"""
        self.data = df
        self._calculate(df)

    def _calculate(self, df):
        """calls internal functions to show headers, descriptive stats and nulls"""
        self._heading(df)
        self._stats(df)
        self._nulls(df)

    def _heading(self, df):
        """Using pandas library to get a glimpse of the dataset"""
        self.lines = df.head()
        self.inf = df.info()
        self.matrix = df.shape
        self.cols = df.columns
    
    def _stats(self, df):
        """Displays all the descriptive stats of the dataset"""
        self.description=df.describe()
        self.relation=df.corr()

        
    def _nulls(self, df):
        """Inspects the whole dataset and returns a list of columns and the respective number of Null values"""
        self.missing = df.isnull().sum()
        """Inspects the whole data frame and replaces any blanks identified earlier with SQLite with NaNs. The NaNs can then be preproessed on the basis of column."""
        self.cl = df.replace(r'^\s*$', np.nan, regex=True)


# In[5]:


#Instantiating an object for the training dataset from the Load class
train = Load(Titanic_Train_table)

# Calling the functions of the Load Class
train._heading(Titanic_Train_table)
train._nulls(Titanic_Train_table)
train._stats(Titanic_Train_table)


# In[6]:


# Having look at the data stored in the attributes
train.lines
#Glimpse of the data


# In[7]:


train.description  
#The description shows a high number of blanks for Age and Cabin, as identified with SQL Queries earlier"""


# In[8]:


train.missing       

#There are no nulls or NAs, but there are blank cells, which we replace with NaNs using regex, defined in our class Load"""


# In[9]:


train.cl            
#Returns a data frame with all the blank cells earlier identified replaced with NaNs.


# In[10]:


#Storing the result of the cleaned data in a new dataframe
train_cleaned = train.cl

#Converting the dtypes of the values in dataframe to numeric
train_cleaned[["Age", "Fare","Survived","SibSp","Parch","Pclass"]] = train_cleaned[["Age", "Fare","Survived","SibSp","Parch","Pclass"]].apply(pd.to_numeric)

#Performing quality checks: if there are values in age column above 100, convert them to NaN.
train_cleaned.loc[train_cleaned.Age > 100, 'Age'] = np.nan


# In[11]:


#Check for the most frequently occuring value in a particular column of a dataframe. In Particular Embarked to replace NaNs.
train_cleaned['Embarked'].value_counts()
#Most Common value is S: Southanmpton


# In[12]:


#Replacing the NaNs in Age column by the median Age Value
train_cleaned['Age'].fillna(train_cleaned['Age'].median(), inplace=True)


#Replacing the NaNs in Embarked column by the most common value, i.e. S
commonvalue_embarked="S"
train_cleaned["Embarked"].fillna(commonvalue_embarked, inplace = True) 


# In[13]:


#Quality Check: Check if all the NANs have been replaced and only Cabin Column left. 
train_cleaned.isnull().sum()


# The only column now left with NaNs is Cabin, around 687. The number is large enough in a dataset of 891 rows. Replacing the NaNs with a certain value will bias the data. It is better to omit this column in later stages of data preparation

# **EXPLAINING OUR FINDINGS FOR THE FIRST PART OF THE ANALYSIS.**
# ***
# *a) What analyses we’ve done on the data:*
# 
# **Till now, we have created a Load Class that loads sqlite database file. Inspected the Blanks by querying the database and replaced them with NaNs using Regex.**
# 
# *b) Why we did these analysis:*
# 
# **It is important to understand the baselines of the dataset from the beginning. For instance, everytime we get a dataset to explore, it is necessary to perform a basic quality check assurance.**

# ### 2.2 Exploring the Data
# <a id="etl"></a>

# In[14]:


class Visuals:
    def __init__(self, df):
        self.data = df
        self._calculate(df)

    def _calculate(self, df):
        """calls internal functions to show basic pandas visualisations"""
        self._visualisations(df)

    def _visualisations(self, df):
        """Display the outliers and the distribution of the data"""
        self.descriptions = df.boxplot(vert = False)
        self.histogram = df.hist(color='k')


# In[15]:


#Instantiating an object of the Visualisation class and inspecting the outliers and data distribution
v_train=Visuals(train_cleaned)


# ### 2.3 Visualisations using seaborn, plotly
# <a id="Visualise"></a>

# In[16]:


#Visualisations using Seaborn


#lm plots using the Logistic model predict the passenger survival
figure1 = sns.lmplot(x="Age", y="Survived", logistic=True, col="Sex",
                         data=train_cleaned, aspect=1, x_jitter=.1, palette="Set1")
    
figure2 = sns.lmplot(x="Fare", y="Survived", logistic=True, col="Sex",
                         data=train_cleaned, aspect=1, x_jitter=.1, palette="Set1")



#Jointplot to understand the Pearson Corelation coefficient between the variables
     
figure3 = sns.jointplot("Age", "Fare", data=train_cleaned, color='b', kind='reg')
    


# In[17]:


#Using Heatmap to visualise the Pearson corelation coefficient between the different , especially useful to understand 
#impact on the Survived column, due to other independent variables


fig, ax = plt.subplots(figsize=(14,14))         # Sample figsize in inches
sns.heatmap(train_cleaned.corr(), annot=True, linewidths=.5, ax=ax)


# ### Summary and inference of lmplots, JointPlot and Heatmap
# ***
# 
# *a) What analyses we’ve done on the data:*
# 
# **We have used three different plots to bring out the relations between the different variables of the training dataset.**
# 
#    **1. Lmplots: Higher Survival rates obsrved for Females as a function of Age and fare. Older Males are observed to show even lower survival rates.**
#    
#    **2. JointPlot: The correlation betwee Fare and Age is quite low and doesnt provide any clear evidence of Ticket Fare being dependent on Age.**
#    
#    **3. Heatmap: Highest Corelation of 0.26 for (Survived) observed with Age, i.e. higher the Age, greater the chance of Survival.**
# 
# *b) Key Inference:*
# 
#    **There are a few outliers in Fare column, followed by the Age Column, as seen from the Boxplots made using Pandas Library. If the models demonstrate lower accuracy, it might be worth coming back and addressing the Outliers.**
#    

# In[18]:


# Detailed histogram of the training data using plotly

figure5 = px.histogram(train_cleaned, x="Age", y="Fare", color="Survived",
                   marginal="box", # or violin, rug
                  hover_data=train_cleaned.columns)
                     
figure5.show()



#ViolinPlots to better understand distiribution of Fare data as a function of Sex and Survival

figure6 = px.violin(train_cleaned, y="Fare", x="Sex", color="Survived", box=True, points="all",
          hover_data=train_cleaned.columns)
figure6.show()


# In[19]:


# scatter plots using Plotly to visualise the survival for each Parch leval as a function of Age and Fare

figure7 = px.scatter(train_cleaned, x="Fare", y="Age", color='Survived',facet_col="Parch")
figure7.show()


# scatter plots using Plotly to visualise the survival for each Parch leval as a function of Age and Fare

figure8 = px.scatter(train_cleaned, x='Fare', y='Age', color='Survived',
                facet_col='SibSp')
figure8.show()


# In[20]:


# Scatter Plots Using Plotly to understand the survival rate for different classes and as a function of Age and Fare

figure9 = px.scatter(train_cleaned, x='Fare', y='Age', color='Survived',
                facet_col='Pclass')
figure9.show()


# Scatter Plots Using Plotly to understand the survival rate for different Boarding stations and as a function of Age and Fare
figure10 = px.scatter(train_cleaned, x='Fare', y='Age', color='Survived',
                facet_col='Embarked')
figure10.show()


# ### Summary and inferences of Histogram and Categorical plots Using Plotly
# ***
# 
# *a) What kind of categorical plots and analyses we’ve done on the data:*
# 
# **We have used three different types of categorical plots to identify trends between Survived and the Independent Variables.**
# 
# **1. Histogram: Between Age and Fare as a function of Survival rate.**
# 
# **2. ViolinPlot: Between Fare and Sex as a function of Survival rate.**
# 
# **3. ScatterPlots: Between Age and Fare as a function of Parch, SibSp, Embarked and Pclass.**
# 
# *b) Inferences:*
# 
# **Histogram and Violin  Plot:**
# 
# *The highest count of people Survived are in the Age Group 28-30 and Females. Indicates that a strong preference was given to Young Females, when life boats were being taken out*
# 
# 
# **Scatterplots**
# 
# *Lower the number of Parch (Parents and Children) and SibSp (Siblings), higher are chances of survival*
# 
# *Pclass=1 had the highest number of Passenger Survival Rate.*
# 
# *Although, intuitively the boarding station should not have an impact on the survival chances, it is observed that highest number of passengers survived boarded from Southampton.*
# 
# 

# ### 2.4 Feature Engineering
# <a id="Engineer"></a>

# In[21]:


class Engineering:
    def __init__(self, df, var1, var2):
        """Initialising the constructor to save the data"""
        self.data = df
        self.variable1 = var1
        self.variable2 = var2
        self._calculate(df, var1, var2)


    def _calculate(self, df, var1, var2):
        """Calculates all the internal functions defined, i.e. featureadded, featuredivision and featureSubtracted"""
        self._featureadded(var1,var2)
        self._featuredivision(var1,var2)
        self._featuremultiplied(var1,var2)
        self._featuresubtracted(var1,var2)
        
    
    def _featuredivision(self,var1,var2):
        """calculates ratio of given two variables"""
        self.dividedfeature=var1//var2
        
    
    def _featuremultiplied(self,var1,var2):
        """calculates product of given two variables"""
        self.multipliedfeature=var1*var2
        
        
    def _featureadded(self,var1,var2):
        """Calculates the addition of two variables"""
        self.addedfeature=var1+var2
        
        
    def _featuresubtracted(self,var1,var2):
        """calculates the subtraction of two variables"""
        self.subtractedfeature=var1-var2


# In[22]:


#Instantiating two objects of the training data from Engineering class
eng_feature1 = Engineering(train_cleaned,train_cleaned['SibSp'], train_cleaned['Parch'])
eng_feature2 = Engineering(train_cleaned,train_cleaned['Age'], train_cleaned['Pclass'])


#Using addition method from the Engineering class on the first object
eng_feature1._featureadded(train_cleaned['SibSp'], train_cleaned['Parch'])


#Using multiplication method from the Engineering class on the second object
eng_feature2._featuremultiplied(train_cleaned['Age'], train_cleaned['Pclass'])


#Saving the data of above two methods into new columns of the cleaned Train dataset. 
train_cleaned['Family'] = eng_feature1.addedfeature
train_cleaned['AgeandClass'] = eng_feature2.multipliedfeature


# In[23]:


# Inspecting the training dataset after Feature Engineering
train_cleaned


# In[25]:


#Feature Engineering. Creating a flag for a certain colum using List Comprehensions
#train_cleaned['flag']=['Green' if x<3 else 'Red' for x in train_cleaned['Family']]
#train_cleaned


# ### 2.5 Preprocessing dataset for model creation
# <a id="preprocess"></a>

# #### 2.5.1 Creation of dummy variables
# <a id="dummies"></a>

# In[24]:


# Getting Dummy Variables for the categorical variables and inspecting the data
train_cleaned = pd.get_dummies(train_cleaned, columns=['Sex','Embarked'])
train_cleaned.head()


# In[25]:


# Segregating the cleaned training dataset into Features and Target variables

train_target=train_cleaned.iloc[:,1]
train_features=train_cleaned.iloc[:,lambda train_cleaned:[2,4,8,10,11,12,13,14,15,16]]

#Inspecting the features dataset
train_features


# #### 2.5.2 Dealing with Outliers: IQR and Z-score method
# <a id="outliers"></a>

# In[26]:


class Outliers:

    def __init__(self,df):
        self.data=df
        self._calculate(df)
    
    def _calculate(self, df):
        """calls internal functions to calculate outliers using the zscore method and the IQR Method"""
        self._zscore(df)
        self._iqr(df)
        self._showiqr(df)
        
    def _zscore(self, df):
        self.z = np.abs(stats.zscore(df))
        print(self.z)
        
    def _iqr(self, df):
        self.Q1 = df.quantile(0.25)
        self.Q3 = df.quantile(0.75)
        self.IQR = self.Q3 - self.Q1
        print(self.IQR)
    
    def _showiqr(self, df):
        print(df < (self.Q1 - 1.5 * self.IQR)) |(df > (self.Q3 + 1.5 * self.IQR))
        
#https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba


# #### 2.5.3 Implementing all the above changes on test dataset

# In[27]:


#Instantiating an object for the test dataset from the Load class
test = Load(Titanic_Test_table)

# Calling the functions of the Load Class and storing all the values
test._heading(Titanic_Test_table)
test._nulls(Titanic_Test_table)
test._stats(Titanic_Test_table)


# In[28]:


# Having look at the data stored in the attributes
test.description  
#The description shows a high number of blanks for Age and Cabin"""


# In[29]:


test.missing       
#There are no nulls or NAs, but there are blank cells, which we replace with NaNs using regex"""


# In[30]:


test.cl            
#Returned a data frame with all the blank cells replaced with NaNs.


# In[31]:


#Storing the result of the cleaned data in a new dataframe
test_cleaned = test.cl

#Converting the dtypes of the values in dataframe to numeric
test_cleaned[["Age", "Fare","SibSp","Parch","Pclass"]] = test_cleaned[["Age", "Fare","SibSp","Parch","Pclass"]].apply(pd.to_numeric)

#Performing quality checks: if there are values in age column above 100, convert them to NaN.
test_cleaned.loc[test_cleaned.Age > 100, 'Age'] = np.nan


# In[32]:


#Quality Check
test_cleaned.isnull().sum()


# In[33]:


test_cleaned['Age'].fillna(test_cleaned['Age'].median(), inplace=True)

#Replacing the NaNs in Age column by the median Age Value"""

test_cleaned['Fare'].fillna(test_cleaned['Fare'].median(), inplace=True)


# In[34]:


#Instantiating two objects of test dataset from the Engineering class
eng_feature1 = Engineering(test_cleaned,test_cleaned['SibSp'],test_cleaned['Parch'])
eng_feature2 = Engineering(test_cleaned,test_cleaned['Age'],test_cleaned['Pclass'])

#Using addition method from the Engineering class on the first object
eng_feature1._featureadded(test_cleaned['SibSp'],test_cleaned['Parch'])

#Using multiplication method from the Engineering class on the second object
eng_feature2._featuremultiplied(test_cleaned['Age'],test_cleaned['Pclass'])

#Saving the data of above two methods into new columns of the original dataset. 
test_cleaned['Family'] = eng_feature1.addedfeature
test_cleaned['AgeandClass'] = eng_feature2.multipliedfeature


# In[35]:


# Getting Dummy Variables for the categorical variables in the test dataset

test_cleaned = pd.get_dummies(test_cleaned, columns=['Sex','Embarked'])
test_cleaned.head()


# In[36]:


# Removing the un-important columns from the test dataset and making it exactly similar to the train dataset.

test_features=test_cleaned.iloc[:,lambda test_cleaned:[1,3,7,9,10,11,12,13,14,15]]

# Inspecting the cleaned test dataset features
test_features


# ### 3: DEVELOP
# <a id="BaselineModel"></a>

# ### 3.1 Model Development
# <a id="development"></a>

# ***Hypothesis***
# 
# ***
# 
# **1. To predict the Survival of Titanic Passengers, we will start by implementing a baseline Logistic Regression model. Without any complexities, it will help to judge how to tune further higher order models.**
# 
# **2. In the next steps, we make use of ensemble, boosting and neighbours algorithms to tune the predictions.**

# In[37]:


class Modelling:
    def __init__(self,var1,var2):
        """Initialising the constructor method to save the data"""
        self.X=var1
        self.Y=var2
        self._calculate(var1, var2)
        
    def _calculate(self, var1, var2):
        """Calculates all the internal functions that contain the different models"""
        self._baselinemodel(var1,var2)
        self._forest(var1,var2)
        self._boosting(var1,var2)
        self._neighbours(var1,var2)
        self._vectors(var1,var2)
        
    def _baselinemodel(self,var1,var2):
        """We start by building a simple logistic regression and then improve upon the results"""
        self.model=LogisticRegression(random_state=42)
        self.model_1=self.model.fit(var1,var2)
        
    def _forest(self,var1,var2):
        """Use the ensemble methods to improve on the baseline model"""
        self.model=RandomForestClassifier(n_estimators=10, max_depth=7, random_state=42)
        self.model_2=self.model.fit(var1,var2)
        
    def _boosting(self,var1,var2):
        """Implemetation of Boosting algorithm to understand differences from the ensemble ones"""
        self.model=GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.2, loss='deviance', max_depth=10,
              max_features=0.3, max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=9, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=10,
              presort='auto', random_state=42, subsample=1.0, verbose=0,
              warm_start=False)
        self.model_3=self.model.fit(var1,var2)
        
    def _neighbours(self,var1,var2):
        """A simple model based on nearest neighbours methodology"""
        self.model=KNeighborsClassifier(n_neighbors = 7)
        self.model_4=self.model.fit(var1,var2)
        
    def _vectors(self,var1,var2):
        """implementing support vector machines"""
        self.model=SVC(kernel = 'linear', C = 1)
        self.model_5=self.model.fit(var1,var2)


# In[38]:


#Instantiating an object of the Modelling class
mod=Modelling(train_features, train_target)

# Using the calculate function to Implement all the models
mod._calculate(train_features, train_target)


#Saving the results of all the models in different variables
lr=mod.model_1
rf=mod.model_2
gb=mod.model_3
kn=mod.model_4
sv=mod.model_5


# ### 3.2 Model Evaluation
# <a id="evaluate"></a>

# In[39]:


class Eval:
    def __init__(self, v1, v2, v3, v4):
        """Initialising the constructor and saving all the variable data"""
        self.X_test=v1
        self.X_train=v2
        self.model=v3
        self.Y_train=v4
        self._calculate(v1, v2, v3, v4)
        
    def _calculate(self, v1, v2, v3, v4):
        self._pred(v1,v3)
        self._cv(v3, v2, v4)
        
    def _pred(self, v1, v3):
        """Predicting the target variable"""
        self.predicted_value=v3.predict(v1)
        
    def _cv(self, v3, v2, v4):
        """Cross validation"""
        self.scores=cross_val_score(v3, v2, v4, cv=4)


# In[40]:


#Instantiating objects for the five different models
eval1=Eval(test_features, train_features, lr, train_target)
eval2=Eval(test_features, train_features, rf, train_target)
eval3=Eval(test_features, train_features, gb, train_target)
eval4=Eval(test_features, train_features, kn, train_target)
eval5=Eval(test_features, train_features, sv, train_target)


# Calculating the cross-validation scores as well as prediction values for all the models
eval1._calculate(test_features, train_features, lr, train_target)
eval2._calculate(test_features, train_features, rf, train_target)
eval3._calculate(test_features, train_features, gb, train_target)
eval4._calculate(test_features, train_features, kn, train_target)
eval5._calculate(test_features, train_features, sv, train_target)

#Storing the cross validation scores of all the models in different variables
scores_lr = eval1.scores
scores_rf = eval2.scores
scores_gb = eval3.scores
scores_kn = eval4.scores
scores_sv = eval5.scores


# Creating a list of all the Cross validation scores and plotting them
cv_scores=[scores_lr, scores_rf, scores_gb, scores_kn, scores_sv]

for x in cv_scores:
    print(x.mean())
    plt.plot(x)
    

# Inspecting the feature importance from the random Forest Model (Prescriptive Statistics)
rf.feature_importances_


#Plottng the Feature Importances
feature_importances = pd.Series(rf.feature_importances_, index=train_features.columns)
print(feature_importances)
feature_importances.sort_values(inplace=True)
feature_importances.plot(kind='barh', figsize=(7,6))


# ### Summary and inferences of Model Development and Model Evaluation
# ***
# 
# *a) What kind of models and evaluation we have performd on the dataset till now?:*
# 
# **We have built a baseline model (Logistic Regression), ensemble models like random Forest, gradient boosting. KNN and SVM.**
# 
# 
# *b) Inferences:*
# 
# **Highest Accuracy using Cross Validation:**
# 
# *Gradient Boosting Classifier outputs the highest mean accuracy of 0.82 for the Titanic Dataset, using a four fold cross validation technique.*
# 
# 
# **Feature Importances**
# 
# *The Feature Importnaces calulcated for the Random Forest model shows Sex (Male) with the highest value and Embarked_Q with lowest values*
# 
# *In the next section, we should work on tuning the Gradient Boosting Classifier and achive better accuracy rate with it.*
# 
# 

# ### 3.3 ModelTuning
# <a id="tuning"></a>

# In[41]:


# Hyper parameter Tuning Using Grid Search
# Define Parameters for Gradient Boosting Model as it shows the best cv score

param_grid = {"max_depth": [2,3,7],
              "max_features" : [1.0,0.3,0.1],
              "min_samples_leaf" : [3,5,9],
              "n_estimators": [8,10,25,50],
              "learning_rate": [0.05,0.1,0.02,0.2]}


# Perform Grid Search CV
from sklearn.model_selection import GridSearchCV
gb_cv = GridSearchCV(gb, param_grid=param_grid, cv = 4, verbose=10, n_jobs=-1 ).fit(train_features, train_target)



# Best hyperparmeter setting
gb_cv.best_estimator_


# ### Summary of Model Tuning
# ***
# 
# *a) How did GridSearch CV work and how to incorporate the parametres back in the model?*
# 
# ****
# 
# **1. With n_estimators=50 and a max_depth of 7, we can try and observe how does the accuracy of GB model change**
# 
# 

# ### 4: DEPLOY
# <a id="deploy"></a>

# In[42]:


# Selecting the best model results out of all the models, (Gradient Boosting)

Best_prediction_result=pd.DataFrame(eval3.predicted_value)
PassID=pd.DataFrame(test_cleaned['PassengerId'])
Submission_file=pd.concat([PassID, Best_prediction_result], axis = 1)
Submission_file.columns=['PassengerId', 'Survived']

def saveresults(df):
    df.to_csv('C:/Users/ishan/Documents/Python Scripts/DSDJ/Best_prediction_Titanic.csv', index=False)


# In[43]:


saveresults(Submission_file)

