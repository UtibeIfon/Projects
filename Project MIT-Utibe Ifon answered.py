#!/usr/bin/env python
# coding: utf-8

# In[5]:


# import libraries for data manipulation
import numpy as np
import pandas as pd

# import libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


#library to suppress warnings
import warnings 
warnings.filterwarnings('ignore')


# In[7]:


# read the data
df = pd.read_csv(r'C:\Users\utyif\.ipynb_checkpoints\foodhub_order.csv')

# this code returns the first 5 rows of the dataset
df.head()


# ## Observations:
# The DataFrame has 9 columns as mentioned in the Data Dictionary. Data in each row corresponds to the order placed by a customer.

# In[4]:


# This code prints the number of rows and columns present in the dataset
rows_columns=df.shape
print('The rows and columns are:', rows_columns, "respectively.")


# The dataset has 1898 rows and 9 columns.

# In[5]:


# this code prints a summary of the DataFrame
df.info()


# ### Observations:
# All columns have the same number of entries, showing that there are all non-null values in them.
# 

# In[6]:


# This code is to check for missing values
df.isna().sum()


# ### Observations:
# There are no missing values from any of the variables. This makes sense as it was earlier observed that all variables have the same number of entries.

# In[10]:


# this code gives a summary of the numerical values in the dataset
df.describe().T


# The minimum time for food to be prepared is 20 minutes.	
# The maximum time for food to be prepared is 35 minutes.
# The average time for food to be prepared is 27.37 minutes.

# In[8]:


df['rating'].value_counts()


# ### Observations:
# Number of orders not rated = 736

# # Exploratory Data Analysis (EDA)

# ### Univariate Analysis

# We will start by exploring and analyzing the numerical variables i.e order_id, customer_id,cost_of_order, food_preparation_time and delivery_time .
# 
# 

# In[15]:


def histogram_boxplot(feature, figsize=(12, 7), bins="auto"):
    """ Boxplot and histogram combined
    
    figsize: size of fig (default (12, 7))
    number of bins set to "auto")
    """
    f, (ax_box, ax_hist) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid
        sharex=True,  
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize
    )

    # Creating the subplots
    #for the boxplot
    sns.boxplot(x=feature, ax=ax_box, showmeans=True, color='violet')

    # For histogram
    sns.histplot(x=feature, kde=False, ax=ax_hist, bins=bins,palette='Paired')
    ax_hist.axvline(np.mean(feature), color='g', linestyle='--')      # this line to add mean to the histogram
    ax_hist.axvline(np.median(feature), color='black', linestyle='-') # this line to Add median

    plt.show()


# ### Order ID

# Let's find out how many unique order IDs exist in the dataset

# In[16]:


#this code shows the number of unique order IDs
print('There are',df['order_id'].nunique(),'unique order IDs')


# ### Customer ID

# In[44]:


#this code shows the number of unique customer IDs
print('There are',df['customer_id'].nunique(),'unique Customer IDs')


# ### Cost of Order

# In[32]:


histogram_boxplot(df.cost_of_the_order)


# ### Observations:
# The minimum cost of an order is 4.47 dollars while the maximum is 35.41 dollars.
# 
# The average cost of an order is about 16.5 dollars.

# ### Food Preparation Time

# In[42]:


histogram_boxplot(df.food_preparation_time)


# ### Observations:
# The minimum preparation time for the orders is 20 minutes and the maximum is 35 minutes.
# 
# The average preparation time is 27.37 minutes.

# ### Delivery Time 

# In[40]:


histogram_boxplot(df.delivery_time)


# ### Observations:
# The minimum delivery time is 15 minutes and the maximum delivery time is 33 minutes.
# The average delivery time is 24.16 minutes.

# ### Now we explore and analyze the categorical variables i.e Restaurant_name, Cuisine_type, day_of_the_week, rating.
# 
# 

# In[17]:


def plot_s(data, f):
    sns.set_style('whitegrid')
    total = len(data[f]) #the length of the column)
    plt.figure(figsize=(15,5))
    
    ax = sns.countplot(data=data, x= f, palette="Paired",order=data[f].value_counts().index)
    
    for p in ax.patches:
            count =  p.get_height() # finds the count of each class of the category
            x = p.get_x() + p.get_width()/2-0.05  # width of the plot
            y = p.get_height()# height of the plot
            ax.annotate(count,(x, y),ha='center', va='center',size=12, xytext=(0,5), textcoords='offset points',)
    plt.show()


# ### Restaurant name

# Let's find out how many unique restaurant names exist in the dataset

# In[17]:


# this code finds out the number of unique restaurant names there are in the dataset
unique_restaurant_name = df['restaurant_name'].nunique()  
print('There are', unique_restaurant_name, 'unique restaurant names.')


# ### Cuisine Type

# We make a count plot to show the cuisine types and their counts 

# In[11]:


#returns a plot to show the different cusine types and their counts
plot_s(df,'cuisine_type')


# We check for how many unique cuisine types there are in the plot

# In[12]:


#this code shows the number of unique cuisine types there are
unique_cuisine_type = df['cuisine_type'].nunique()
print('There are',unique_cuisine_type, "unique cuisine types as seen in the above plot.")


# ### Observations:
# We can see that from the 14 unique cuisine types, American is the most ordered cuisine type with 584 orders, making it the most popular.
# 
# Second most popular is Japanese with 470.
# 
# The least ordered cuisine type is Vietnamese with only 7 orders.

# ### Day of the Week

# In[76]:


plot_s(df, 'day_of_the_week')


# ### Observation:
# The number of orders for weekend greatly exceeds that of weekday, this shows that people prefer to order out on weekends.

# ### Rating

# In[48]:


#this code shows the unique categories for Rating
print('There are',df['rating'].nunique(),'unique categories for rating, they are:',
      df['rating'].unique())


# In[77]:


#this code displays the ratings and their counts
plot_s(df, 'rating')


# ### Observation:
# The distribution shows that 736 orders did not get a rating.
# 
# 588 orders received the highest rating of 5.
# 
# 188 orders received the lowest  rating of 3.

# ### Top 5 restaurants in terms of the number of orders received

# In[47]:


#top 5 restaurants in terms of the number of orders received
df['restaurant_name'].value_counts().sort_values(ascending = False).head(5)


# ### Observation:
# Shake Shack receives the highest number of orders, 219 orders.
# 
# The Meatball shop and The Blue Ribbon Sushi are 2nd and 3rd respictively with 132 and 119 orders for each.
# 

# ### Most popular cuisine on the weekends

# In[58]:


#most popular cuisine on the weekends
df_weekend = df[df['day_of_the_week'] == 'Weekend']
df_weekend['cuisine_type'].value_counts().sort_values(ascending=False).head(1)


# #### Observations:
# Most popular cuisine on weekends is American Cuisine.

# ### Percentage of the orders that cost more than 20 dollars:

# In[123]:


#percentage of the orders that cost more $20
# code for orders greater than $20
orders_greater_than_20 = df[df['cost_of_the_order']>20] 
orders_greater_than_20.shape[0]

#  code for percentage of orders greater than $20
percentage = (orders_greater_than_20.shape[0] / df.shape[0]) * 100

print("Percentage of orders that cost more than $20 dollars =", round(percentage, 2), '%') #rounded to 2 decimal places


# ### Mean order delivery time :

# In[150]:


# code for mean delivery time 
mean_delivery_time = df['delivery_time'].mean()
print("The mean order delivery time is :" ,round(mean_delivery_time,2),"minutes") #rounded to 2 decimal places


# ### Top 3 most frequent customers

# In[125]:


#we get the most frequent customers by getting the top 3 customer_ids
df['customer_id'].value_counts().head(3)


# #### Observations:
# The top 3 most frequent customers are customers with customer id: 52832, 47440 and 83287.
# These are the customers that will get the 20% discount voucher.

# ## Multivariate Analysis

# #### We will check for correlation amongst the numerical variables

# In[8]:


numerical_variables = ['order_id', 'customer_id',
                       'cost_of_the_order', 'food_preparation_time','delivery_time']
corr = df[numerical_variables].corr()

# for the heatmap
plt.figure(figsize = (12, 7))
sns.heatmap(corr, annot = True, vmin=-0.7, vmax=1, cmap = 'coolwarm',            
        fmt = ".1f",            
        xticklabels = corr.columns,            
        yticklabels = corr.columns)


# ### Observation:
# There is no correlation between the cost, preparation time and the delivery time.

# ### Cuisine vs. Cost of Order

# We will analyze the relationship between the cuisine and the cost of the order

# In[126]:


#code to plot a boxplot
plt.figure(figsize=(15,7))
sns.boxplot(x = "cuisine_type", y = "cost_of_the_order", data = df, palette = 'Blues')
plt.xticks(rotation = 60)
plt.show()


# ### Observations:
# Vietnamese and Korean cusines are the least costing cuisines.
# 
# There are outliers present for the cost of Korean, Mediterranean and Vietnamese cuisines.
# 
# French cuisine seems to be the most expensive cuisine.

# ### Cuisine VS Preparation time
# 

# In[92]:


# boxplt to show relationship between food preparation time and cuisine type
plt.figure(figsize=(15,7))
sns.boxplot(x='cuisine_type',y='food_preparation_time', data=df, palette="Paired")
plt.xticks(rotation = 60)
plt.show()


# ### Observations:
# Most cuisine types have around the same preparation times.
# 
# Outliers are present for the food preparation time of Korean cuisine.
# 
# Korean cuisine has the least preparation time.
# 

# ### Day of the Week vs Delivery Time

# In[108]:


#we will visualize the relationship between day of the week and delivery time using a boxplot
sns.set_style('whitegrid')
plt.figure(figsize=(15,7))
sns.boxplot(x='day_of_the_week',y='delivery_time', data= df, palette='Blues')  
plt.show()


# ### Observation:
# Delivery time for most of the weekend orders is less than that of weekdays.

# ### Day of the Week and Cuisine Type

# In[101]:


sns.swarmplot(data=df, x='day_of_the_week',y='cuisine_type', palette='Paired')
plt.grid()
plt.xticks(rotation=360)


# ### Observations:
# Orders received on Weekends are more than that of weekdays for all cusine types.
# 
# Orders received for Spanish, Korean,Thai and French on Weekdays are significantly less than orders received on weekends.
# 
# Japenese, Mexican, American, Indian, Italian, Mediterranean Chinese and Middle Eastern have equal amounts of orders on Weekends, comparative difference  exists only on weekdays.
# 
# 

# ### Rating vs Food Preparation time

# In[69]:


# Relationship between rating and food preparation time using a pointplot
sns.set_style('whitegrid')
plt.figure(figsize=(15, 7))
sns.pointplot(x='rating', y='food_preparation_time', data= df)  
plt.show()


# ### Observation:
# For the rated orders, orders with preparation time 27.4 minutes or below got a rating of 4 or higher.
# 

# ### Rating and Delivery Time

# In[105]:


sns.set_style('darkgrid')
plt.figure(figsize=(15,7))
sns.axes_style('darkgrid')
sns.pointplot(data= df,x='rating', y='delivery_time')
plt.show()


# ### Observation:
# For the orders that were rated, delivery times below 24.25 minutes were rated a rating of 4 or higher.
# 
# It is possible that delivery time may play a role in the low rating of orders.

# ### Rating vs Cost of Order

# In[106]:


plt.figure(figsize=(15, 7))
sns.pointplot(data=df, x='rating',y='cost_of_the_order')   
plt.show()


# ### Observations:
# For the rated orders, orders that cost more than 16.5 dollars were given a rating of 4 or higher.
# 
# Orders below 16.5 dollars were given a rating of 3 or unrated.
# 
# Higher costing orders have higher ratings while lower costing orders are rated lowly or unrated.

# #### Promotinal offer for rated restaurants

# In[13]:


#first filter the rated restaurants
rated_restaurants = df[df['rating']!='Not given'].copy()
#change the data type to int
rated_restaurants['rating'] = rated_restaurants['rating'].astype('int')

#new dataframe with the names and count
rated_restaurants_count = rated_restaurants.groupby(['restaurant_name'])['rating'].count().sort_values(ascending=False).reset_index()
rated_restaurants_count.head()



# In[14]:


#restaurant names with rating >50

names_50 = rated_restaurants_count[rated_restaurants_count['rating']>50]['restaurant_name']

# Filter to get the data of restaurants that have rating count more than 50
filter_1 = rated_restaurants[rated_restaurants['restaurant_name'].isin(names_50)].copy()

#the restaurant names with their individual mean rating
filter_1.groupby(['restaurant_name'])['rating'].mean().sort_values(ascending = False).reset_index().dropna() 


# There are 4 restaurants fulfilling the criteria for the promational offer.

# ### Net Revenue

# In[133]:


#revenue function
def net_revenue(n):
    if n > 20:
        return n*0.25
    elif n > 5:
        return n*0.15
    else:
        return n *0
    
df['Revenue']= df['cost_of_the_order'].apply(net_revenue)
df.head()


# In[140]:


total_rev = df['Revenue'].sum() 
print('The net revenue is $', round(total_rev, 2))


# #### Orders that take more than 60 minutes to get delivered

# In[144]:


#new column called total_time
df['total_time'] = df['food_preparation_time'] +df['delivery_time']
#percentage of orders > 60 minutes total delivery time
print("Percentage of orders with more than 60 minutes total delivery time is:",
      round(df[df['total_time']>60].shape[0]/df.shape[0] *100, 2), '%')


# #### Mean delivery between weekends and weekdays

# In[145]:


##mean delivery for weekdays
print('The mean delivery time on weekdays is', 
      round(df[df['day_of_the_week'] == 'Weekday']['delivery_time'].mean()),
     'minutes')


# In[148]:


##mean delivery for weekends
print('The mean delivery time on weekend is',round(df[df['day_of_the_week']=='Weekend']['delivery_time'].mean()),'minutes')


# ## Conclusions and Recommendations

# ### Conclusions:
# The top 3 most popular cusines: American, Japanese and Italian account for about 70% of the total orders. This shows that these cuisines are quite preferable to the FoodHub customers.
# 
# Shake Shack restaurant has had the highest number of orders, and consequentially the highest number of ratings.
# 
# The Meatball Shop is the restaurant with the highest average rating (4.51)
# 
# The amount of orders more than double during the weekend. This shows Foodhub customers prefer to order on weekends than weekdays.
# 
# There are longer delivery times during the weekdays, and reduced delivery times on the weekends.
# 

# ### Recommendations:
# Foodhub should collaborate with the restaurants that serve American, Japanese and Italian cuisine. These are the 3 most popular amongst Foodhub customers and thus might be a profitable venture for the company.
# 
# The number of orders not rated by customers is 736, which amounts to about 38.8 percent of total orders. Rating is an important way to tell the level of customer satisfaction. Therefore the company should investigate why so many orders don't get rated, and maybe incentivise customers to rate their orders.
# 
# 
# There is a clear preference for ordering on the weekend, the company should take advantage of this by making more delivery personnel to be available on weekends so as to meet up with this increasing demand.
