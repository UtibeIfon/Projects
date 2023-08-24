#!/usr/bin/env python
# coding: utf-8

# # ExtraaLearn Project
# 
# ## Context
# 
# The EdTech industry has been surging in the past decade immensely, and according to a forecast, the Online Education market would be worth $286.62bn by 2023 with a compound annual growth rate (CAGR) of 10.26% from 2018 to 2023. The modern era of online education has enforced a lot in its growth and expansion beyond any limit. Due to having many dominant features like ease of information sharing, personalized learning experience, transparency of assessment, etc, it is now preferable to traditional education. 
# 
# In the present scenario due to the Covid-19, the online education sector has witnessed rapid growth and is attracting a lot of new customers. Due to this rapid growth, many new companies have emerged in this industry. With the availability and ease of use of digital marketing resources, companies can reach out to a wider audience with their offerings. The customers who show interest in these offerings are termed as leads. There are various sources of obtaining leads for Edtech companies, like
# 
# * The customer interacts with the marketing front on social media or other online platforms. 
# * The customer browses the website/app and downloads the brochure
# * The customer connects through emails for more information.
# 
# The company then nurtures these leads and tries to convert them to paid customers. For this, the representative from the organization connects with the lead on call or through email to share further details.
# 
# ## Objective
# 
# ExtraaLearn is an initial stage startup that offers programs on cutting-edge technologies to students and professionals to help them upskill/reskill. With a large number of leads being generated on a regular basis, one of the issues faced by ExtraaLearn is to identify which of the leads are more likely to convert so that they can allocate resources accordingly. You, as a data scientist at ExtraaLearn, have been provided the leads data to:
# * Analyze and build an ML model to help identify which leads are more likely to convert to paid customers, 
# * Find the factors driving the lead conversion process
# * Create a profile of the leads which are likely to convert
# 
# 
# ## Data Description
# 
# The data contains the different attributes of leads and their interaction details with ExtraaLearn. The detailed data dictionary is given below.
# 
# 
# **Data Dictionary**
# * ID: ID of the lead
# * age: Age of the lead
# * current_occupation: Current occupation of the lead. Values include 'Professional','Unemployed',and 'Student'
# * first_interaction: How did the lead first interacted with ExtraaLearn. Values include 'Website', 'Mobile App'
# * profile_completed: What percentage of profile has been filled by the lead on the website/mobile app. Values include Low - (0-50%), Medium - (50-75%), High (75-100%)
# * website_visits: How many times has a lead visited the website
# * time_spent_on_website: Total time spent on the website
# * page_views_per_visit: Average number of pages on the website viewed during the visits.
# * last_activity: Last interaction between the lead and ExtraaLearn. 
#     * Email Activity: Seeking for details about program through email, Representative shared information with lead like brochure of program , etc 
#     * Phone Activity: Had a Phone Conversation with representative, Had conversation over SMS with representative, etc
#     * Website Activity: Interacted on live chat with representative, Updated profile on website, etc
# 
# * print_media_type1: Flag indicating whether the lead had seen the ad of ExtraaLearn in the Newspaper.
# * print_media_type2: Flag indicating whether the lead had seen the ad of ExtraaLearn in the Magazine.
# * digital_media: Flag indicating whether the lead had seen the ad of ExtraaLearn on the digital platforms.
# * educational_channels: Flag indicating whether the lead had heard about ExtraaLearn in the education channels like online forums, discussion threads, educational websites, etc.
# * referral: Flag indicating whether the lead had heard about ExtraaLearn through reference.
# * status: Flag indicating whether the lead was converted to a paid customer or not.

# ## Importing necessary libraries and data

# In[2]:


import warnings

warnings.filterwarnings("ignore")
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)

# Libraries to help with reading and manipulating data

import pandas as pd
import numpy as np

# Library to split data
from sklearn.model_selection import train_test_split

# libaries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Removes the limit for the number of displayed columns
pd.set_option("display.max_columns", None)
# Sets the limit for the number of displayed rows
pd.set_option("display.max_rows", 200)
# setting the precision of floating numbers to 5 decimal points
pd.set_option("display.float_format", lambda x: "%.5f" % x)

# To build model for prediction
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# To tune different models
from sklearn.model_selection import GridSearchCV


# To get diferent metric scores
import sklearn.metrics as metrics
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    make_scorer,
)


# In[4]:


#import dataset
learn = pd.read_csv(r'C:\Users\utyif\OneDrive\Desktop\Courses\MIT IDSS DSML\PROJECT 2\ExtraaLearn.csv')
data = learn.copy()


# ## Data Overview
# 
# - Observations
# - Sanity checks

# In[5]:


#first 5 rows of the dataset
data.head()


# In[6]:


#last five rows of the dataset
data.tail()


# In[8]:


#the shape of the dataset
data.shape


# #### the dataset has 4612 rows and 15 columns

# In[9]:


#data type 
data.info()


# ### Observation:
# age, website_visits, time_spent_on_website, page_views_per_visit, and status are numeric datatypes while rest of the columns are object data types

# In[10]:


#checking for duplicates
data.duplicated().sum()

There are no duplicate values in the data
# ## Exploratory Data Analysis (EDA)
# 
# - EDA is an important part of any project involving data.
# - It is important to investigate and understand the data better before building a model with it.
# - A few questions have been mentioned below which will help you approach the analysis in the right manner and generate insights from the data.
# - A thorough analysis of the data, in addition to the questions mentioned below, should be done.

# # **Questions**
# 1. Leads will have different expectations from the outcome of the course and the current occupation may play a key role in getting them to participate in the program. Find out how current occupation affects lead status.
# 2. The company's first impression on the customer must have an impact. Do the first channels of interaction have an impact on the lead status? 
# 3. The company uses multiple modes to interact with prospects. Which way of interaction works best? 
# 4. The company gets leads from various channels such as print media, digital media, referrals, etc. Which of these channels have the highest lead conversion rate?
# 5. People browsing the website or mobile application are generally required to create a profile by sharing their personal data before they can access additional information.Does having more details about a prospect increase the chances of conversion?

# ## Data Preprocessing
# 
# - Missing value treatment (if needed)
# - Feature engineering (if needed)
# - Outlier detection and treatment (if needed)
# - Preparing data for modeling 
# - Any other preprocessing steps (if needed)

# ## EDA
# 
# - It is a good idea to explore the data once again after manipulating it.

# In[11]:


#statistical summary of the data
data.describe()


# ### Observations:
# Mean age of approximately 46.2
# Mean website vists is approximately 3.5, this is a low number of site visits.
# Mean of status is 0.29857, this means that 29.85% of leads are converted.

# In[14]:


#list of categorical variables
cat_col = list(data.select_dtypes("object").columns)

for column in cat_col:
    print(data[column].value_counts())
    print("-" * 50)


# In[15]:


#unique variables
data["ID"].nunique()


# All the values in the ID column are unique
# this column can be dropped , it doesnt neccessarily add value to our analysis

# In[16]:


#drop column
data.drop(['ID'], axis=1, inplace= True)


# ## Univariate analysis
# we will analyze the univariate variables first. These include age, website_visits, time_spent_on_website, page_views_per_visit,...etc

# In[17]:


# we will plot a boxplot and a histogram

def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
   
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a star will indicate the mean value of the column
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram


# ## Age:

# In[18]:


histogram_boxplot(data, 'age')


# In[ ]:





# ## Website_visits
# 

# In[19]:


histogram_boxplot(data, 'website_visits')


# In[20]:


data[data["website_visits"] == 0].shape


# In[ ]:





# ## Time_spent_on_website

# In[21]:


histogram_boxplot(data, 'time_spent_on_website')


# In[ ]:





# ## Page_views_per_visit

# In[23]:


histogram_boxplot(data, 'page_views_per_visit')


# In[ ]:





# In[26]:


#labeled barplot
def labeled_barplot(data, feature, perc=False, n=None):

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  


# 

# ## Current_occupation

# In[27]:


labeled_barplot(data, "current_occupation", perc=True)


# ## First_interaction

# In[32]:


labeled_barplot(data,'first_interaction', perc= True)


# In[ ]:





# ### Profile_completed

# In[35]:


labeled_barplot(data,'profile_completed', perc= True)


# In[ ]:





# ## Last_activity

# In[36]:


labeled_barplot(data, 'last_activity', perc= True)


# In[ ]:





# ## Print_media_type1

# In[37]:


labeled_barplot(data,'print_media_type1', perc = True)


# In[ ]:





# ## Print_media_type2

# In[38]:


labeled_barplot(data,'print_media_type2', perc = True)


# In[ ]:





# ## digital_media

# In[39]:


labeled_barplot(data,'digital_media', perc = True)


# In[ ]:





# ## educational_channels

# In[40]:


labeled_barplot(data,'educational_channels', perc = True)


# In[ ]:





# ## referral

# In[41]:


labeled_barplot(data,'referral', perc = True)


# In[ ]:





# ## status

# In[42]:


labeled_barplot(data,'status', perc = True)


# ## Bivariate analysis
# 

# In[48]:


plt.figure(figsize = (10, 6))
sns.countplot(x = 'current_occupation', hue = 'status', data = data)
plt.show()


# ### Observations:
# The plot shows that working professional leads are the most likely to be converted leads, and students are least the likely to be converted.

# In[ ]:





# Let us use age as a factor to differentiate

# In[51]:


plt.figure(figsize = (10, 5))
sns.boxplot(x="current_occupation",y ='age', data= data)
plt.show()


# In[52]:


data.groupby(["current_occupation"])["age"].describe()


# ### Observations:
# The range of age for students is 18 to 25 years.
# The range of age for professionals is 25 to 60 years.
# The range of age for unemployed leads is 32 to 63 years.
# The mean age for working professionals and unemployed is almost the same

# In[53]:


#this code to show the relationship between the channels of first interaction and conversion of leads
plt.figure(figsize = (10, 6))
sns.countplot(x = 'first_interaction', hue = 'status', data = data)
plt.show()


# ### Observations
# The website seems to be doing a better job than the mobile app in the terms of lead conversion.
# Leads that interacted through websites were converted more to paidcustomers, than those that interacted through the mobileapp.

# In[ ]:





# #### We will explore the relationship between time spent on websites and conversion

# In[55]:


plt.figure(figsize = (10, 5))
sns.boxplot(x="status",y ='time_spent_on_website', data= data)
plt.show()


# ### Observations:
# Time spent by both non-paying customers and paying customers are roughly the same.
# 
# There are many outliers for non-paying customers.
# 
#     

# #### We will next analyze the relationship between profile completion and lead status

# In[56]:


plt.figure(figsize = (10, 6))
sns.countplot(x = 'profile_completed', hue = 'status', data = data)
plt.show()


# ### Observations:
# The leads whose profile completion level is high converted more in comparison to other levels of profile completion.
# The medium and low levels of profile completion saw comparatively very lessconversions.
# The high level of profile completion might indicate a lead's intent to pursue the course which results in high conversion.

# In[ ]:





# ### Referrals vs Lead Conversion

# In[57]:


plt.figure(figsize = (10, 6))
sns.countplot(x = 'referral', hue = 'status', data = data)
plt.show()


# #### Observations:
# There's a low number of referrals.
# There's is a high conversion rate.

# In[ ]:





# ### Correlation Heatmap

# In[43]:


cols_list = data.select_dtypes(include=np.number).columns.tolist()

plt.figure(figsize=(12, 7))
sns.heatmap(
    data[cols_list].corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral"
)
plt.show()


# ### Observations:
# There is a slight positive correlation between age and status
# time_spent_on_website has a positive correlation with status

# #### Outliers

# In[58]:


# outlier detection using boxplot
numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
# dropping release_year as it is a temporal variable
numeric_columns.remove("status")

plt.figure(figsize=(15, 12))

for i, variable in enumerate(numeric_columns):
    plt.subplot(4, 4, i + 1)
    plt.boxplot(data[variable], whis=1.5)
    plt.tight_layout()
    plt.title(variable)

plt.show()


# ### Observations
# page_views_per_visit has the most outliers in the dataset followed by website_visits
# Age and time_spent_on_website have no outliers
# 

# ### Data preparation for modeling

# In[59]:


X = data.drop(["status"], axis=1)
Y = data['status']

X = pd.get_dummies(X, drop_first=True)

# Splitting the data in 70:30 ratio for train to test data
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.30, random_state=1
)


# In[60]:


print("Shape of Training set : ", X_train.shape)
print("Shape of test set : ", X_test.shape)
print("Percentage of classes in training set:")
print(y_train.value_counts(normalize=True))
print("Percentage of classes in test set:")
print(y_test.value_counts(normalize=True))


# In[ ]:





# ## Building Classification Models

# In[63]:


# Function to print the classification report and get confusion matrix in a proper format

def metrics_score(actual, predicted):
    print(classification_report(actual, predicted))
    
    cm = confusion_matrix(actual, predicted)
    
    plt.figure(figsize = (8, 5))
    
    sns.heatmap(cm, annot = True,  fmt = '.2f', xticklabels = ['Not Converted', 'Converted'], yticklabels = ['Not Converted', 'Converted'])
    
    plt.ylabel('Actual')
    
    plt.xlabel('Predicted')
    
    plt.show()


# ## Building a Decision Tree model

# In[64]:


# Fitting the decision tree classifier on the training data
d_tree =DecisionTreeClassifier(random_state = 7)
d_tree.fit(X_train, y_train)


# In[65]:


#performance on the training data
y_pred_train1 = d_tree.predict(X_train)

metrics_score(y_train, y_pred_train1)


# ### Observations:
# There is less error on the training set, i.e., each sample has been classified correctlyexcept few points. The model has performed very well on the training set

# In[66]:


#performance on the testing data
y_pred_test1 = d_tree.predict(X_test)
metrics_score(y_test, y_pred_test1)


# ### Observations:
# The Decision Tree works better on the training data than on the test data as the recall for class 1 is <0.7 in comparison to 1 for the training dataset. The DecisionTree is overfitting the training data.

# In[ ]:





# ### Decision Tree - Hyperparameter Tuning

# In[67]:


d_tree_tuned = DecisionTreeClassifier(random_state = 7, class_weight = {0: 0.3, 1: 0.7})

# Grid of parameters to choose from
parameters = {'max_depth': np.arange(2, 10), 
              'criterion': ['gini', 'entropy'],
              'min_samples_leaf': [5, 10, 20, 25]
             }

# Type of scoring used to compare parameter combinations 
scorer = metrics.make_scorer(recall_score, pos_label = 1)

# Run the grid search
grid_obj = GridSearchCV(d_tree_tuned, parameters, scoring = scorer, cv = 5)

grid_obj = grid_obj.fit(X_train, y_train)

# Set the classifier to the best combination of parameters
d_tree_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data
d_tree_tuned.fit(X_train, y_train)


# In[ ]:





# In[68]:


# model performance on training data
y_pred_train2 = d_tree_tuned.predict(X_train)
metrics_score(y_train, y_pred_train2)


# ### Observations:
# Performance on training has reduced significantly in comparison to the model with default values.

# In[69]:


#performance on testing data
y_pred_test2 = d_tree_tuned.predict(X_test)
metrics_score(y_test, y_pred_test2)


# ### Observations:
# the tuned model has a higher class 1 recall score than the default model.
# 
# Class 1 precision score has slightly decreases but this holds less importance.
# 
# The model is not overfitting the training data.

# In[ ]:





# ### Visualizing the Decision Tree

# In[70]:


features = list(X.columns)

plt.figure(figsize = (20, 20))

tree.plot_tree(d_tree_tuned, feature_names = features, filled = True, fontsize = 9, node_ids = True, class_names = True)

plt.show()


# Blue represents the converted leads
# Orange represents those not converted
# The color of a leaf gets darker the more observations it has in it.

# ### Observations:
# For leads below 25 years old with first interaction via mobile, if the overall time spent on the website is less than 7 minutes, they are more likely to be not converted leads. 
# 
# For leads over 25 years old with first interaction via mobile, and overall time spent on the website is over 7 minutes, they are more likely to be converted leads.
# 
# 

# ### Feature Importance

# In[71]:


#feature importance
print (pd.DataFrame(d_tree_tuned.feature_importances_, columns = ["Imp"], index = X_train.columns).sort_values(by = 'Imp', ascending = False))


# In[77]:


#plot of feature importance
importances = d_tree_tuned.feature_importances_

indices = np.argsort(importances)

plt.figure(figsize = (10, 10))

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color = 'violet', align = 'center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()


# ### Observations:
# Time spent on the website and first_interaction_website are the most important features followed by profile_completed, age, and last_activity.
# 
# The rest of the variables have no impact in this model, while deciding whether a lead will be converted or not.

# ## Building a Random Forest model

# In[79]:


# Fitting the random forest tree classifier on the training data
rf_estimator = RandomForestClassifier(random_state = 7)

rf_estimator = rf_estimator.fit(X_train, y_train)


# In[80]:


#performance on the training data
y_pred_train3 = rf_estimator.predict(X_train)

metrics_score(y_train, y_pred_train3)


# ### Observations:
# Just like the decision tree, this random forest is giving 100% scores for all the metrics on training data.

# In[3]:


# Checking performance on the testing data
y_pred_test3 = rf_estimator.predict(X_test)

random_forest1 = metrics_score(y_test, y_pred_test3)
random_forest1


# ### Observation
# The tuned decision tree model did a better job of identifying true positives, whereas the forest is doing the best job of predicting true negatives.
# 
# We can conclude that the since tuned decision tree using gridsearch CV has a higher recall for class 1 (0.86) compared to random forest (0.69) the decision tree model is better suited for our purpose.
# 
# The Random Forest classifier seems to be overfitting the training data.

# In[ ]:





# ### Random Forest Classifier - Hyperparameter Tuning

# In[2]:


#classifier
rf_estimator_tuned = RandomForestClassifier(criterion = "entropy", random_state = 7)

# Grid of parameters to choose from
parameters = {"n_estimators": [100, 110, 120],
    "max_depth": [5, 6, 7],
    "max_features": [0.8, 0.9, 1]
             }

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(recall_score, pos_label = 1)

# Run the grid search
grid_obj = GridSearchCV(rf_estimator_tuned, parameters, scoring = scorer, cv = 5)

grid_obj = grid_obj.fit(X_train, y_train)

# Set the classifier to the best combination of parameters
rf_estimator_tuned = grid_obj.best_estimator_


# In[97]:


# Fitting the best algorithm to the training data
rf_estimator_tuned.fit(X_train, y_train)


# In[1]:


# Checking performance on the training data
y_pred_train4 = entropy_tuned.predict(X_train)

metrics_score(y_train, y_pred_train4)


# In[ ]:





# ### Observations
# We can see that after hyperparameter tuning, the model is performing poorly on the train data as well.
# 
# We can try adding some other hyperparameters and/or changing values of some hyperparameters to tune the model and see if we can get better performance.

# In[ ]:





# In[98]:


#feature importance
importances = rf_estimator_tuned.feature_importances_

indices = np.argsort(importances)

feature_names = list(X.columns)

plt.figure(figsize = (12, 12))

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color = 'violet', align = 'center')

plt.yticks(range(len(indices)), [feature_names[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()


# ### Observations:
# Similar to the decision tree model, time spent on website, first_interaction_website, profile_completed, and age are the top four features that help distinguish between not converted and converted leads.
# 
# Unlike the decision tree, the random forest gives some importance to other variables like occupation, page_views_per_visit, as well. This implies that the random forest is giving importance to more factors in comparison to the decision tree.

# ## Actionable Insights and Recommendations

# ### Conclusions:
# The best model we have got so far is the tuned random forest model which is giving 88.9% recall for class 1 on the test data.
# 
# Time spent on website, first_interaction_website,profile_completed, and age seem to be the most important feature.
# 
# Based on the performance of the models, as well as the feature importance, we can safely say that time spent on website, first contact, and how much of the profile is completed are together strong indicators whether or not somoneone will become a converted lead or not.

# In[ ]:





# ### Recommendations:
# It seems the website is working relatively well in terms of converting leads whose firstinteraction is through the website. Thus, the company should try to further increasethe traffic to its websit so that more leads can have their first interaction via website.
# 
# The company should try to reach more working professionsalleads as that segmentappear to have highest conversion rate, probably due to their higher fee-payingcapabilities
