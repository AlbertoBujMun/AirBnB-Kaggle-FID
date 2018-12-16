import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Inspect available files
list_resources_directory = os.listdir('lib')

# Load data
train = pd.read_csv("lib/train_users_2.csv")
# age_gender = pd.read_csv("lib/age_gender_bkts.csv")
# countries = pd.read_csv("lib/countries.csv")
# session = pd.read_csv("lib/sessions.csv")

# Explore data
train.info()
ten_first = train.head(10)
data_types = train.dtypes

# Clean data
"""
    In each step we will be reduce the number of "abnormal" data, getting a better distribution to work with
"""

# Remove id column
train = train.drop(['id'], axis=1)

# Remove the rows where one element or more is/are missing
train.dropna(inplace=True)

# Keep only the year in date type objects
train["date_account_created"] = [int(date.split("-")[0]) for date in train.iloc[:, 0]]
train["date_first_booking"] = [int(each.split("-")[0]) for each in train.iloc[:, 2]]

# Converts the first active timestamp column into the specific year
train["timestamp_first_active_year"] = train["timestamp_first_active"].astype(str).str[0:4]
train = train.drop(['timestamp_first_active'], axis=1)

ten_first = train.head(10)

# ------------------------ Gender study by removing -unknown- gender tuples ------------------------
data = train[(train.gender == "MALE") | (train.gender == "FEMALE")]
ten_first_data = data.head(10)

# Percentage of women and men in the population study by years
years_of_study = data.date_account_created.value_counts()
gender_percentage2010 = data.gender[data.date_account_created == 2010].value_counts() / data.shape[0] * 100
gender_percentage2011 = data.gender[data.date_account_created == 2011].value_counts() / data.shape[0] * 100
gender_percentage2012 = data.gender[data.date_account_created == 2012].value_counts() / data.shape[0] * 100
gender_percentage2013 = data.gender[data.date_account_created == 2013].value_counts() / data.shape[0] * 100
gender_percentage2014 = data.gender[data.date_account_created == 2014].value_counts() / data.shape[0] * 100

print("Gender percentage 2010: \n", gender_percentage2010, "\n")
print("Gender percentage 2011: \n", gender_percentage2011, "\n")
print("Gender percentage 2012: \n", gender_percentage2012, "\n")
print("Gender percentage 2013: \n", gender_percentage2013, "\n")
print("Gender percentage 2014: \n", gender_percentage2014, "\n")

# ------------------------ Destination study by years ------------------------
x2010_values = data.country_destination[data.date_account_created == 2010].value_counts()
x2011_values = data.country_destination[data.date_account_created == 2011].value_counts()
x2012_values = data.country_destination[data.date_account_created == 2012].value_counts()
x2013_values = data.country_destination[data.date_account_created == 2013].value_counts()
x2014_values = data.country_destination[data.date_account_created == 2014].value_counts()

# ------------------------ Distribution by age per year ------------------------
x2010 = data[data.date_account_created == 2010]
x2011 = data[data.date_account_created == 2011]
x2012 = data[data.date_account_created == 2012]
x2013 = data[data.date_account_created == 2013]
x2014 = data[data.date_account_created == 2014]

plt.figure()
plt.boxplot([x2010.age, x2011.age, x2012.age, x2013.age, x2014.age])
plt.savefig('charts and diagrams/agesdistribution.png')

# ------------------------ Removing the upper outliers ------------------------

p1 = np.percentile(x2010.age, [5, 95])
p2 = np.percentile(x2011.age, [5, 95])
p3 = np.percentile(x2012.age, [5, 95])
p4 = np.percentile(x2013.age, [5, 95])
p5 = np.percentile(x2014.age, [5, 95])

new_df1 = x2010[(x2010.age > p1[0]) & (x2010.age < p1[1])]
new_df2 = x2011[(x2011.age > p2[0]) & (x2011.age < p2[1])]
new_df3 = x2012[(x2012.age > p3[0]) & (x2012.age < p3[1])]
new_df4 = x2013[(x2013.age > p4[0]) & (x2013.age < p4[1])]
new_df5 = x2014[(x2014.age > p5[0]) & (x2014.age < p5[1])]

plt.figure()
plt.boxplot([new_df1.age, new_df2.age, new_df3.age, new_df4.age, new_df5.age])
plt.savefig('charts and diagrams/agesdistribution_without_outliers.png')