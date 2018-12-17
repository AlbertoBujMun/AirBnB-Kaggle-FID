import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import stats

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
gender_distribution = sns.countplot(train['gender'], palette="deep")
plt.savefig('charts and diagrams/genderdistribution.png')

# there is any trend in destinations in users who put unknown as their gender?
unknown_gender = train.loc[train.gender == '-unknown-', 'country_destination'].value_counts().sum()
unknown_gender_countries = train.loc[train.gender == '-unknown-', 'country_destination'].value_counts() / (
        100 * unknown_gender)

known_gender = train.loc[train.gender != '-unknown-', 'country_destination'].value_counts().sum()
known_gender_countries = train.loc[train.gender != '-unknown-', 'country_destination'].value_counts() / (
        100 * unknown_gender)

compared_known_and_unknown_genders = pd.concat([unknown_gender_countries, known_gender_countries], axis=1)
compared_known_and_unknown_genders.columns = ['unknown_gender countries', 'known_genders countries']

# So, individuals who do not indicate gender, did not book a trip

# working with clean gender dataset
# there is any trend in destinations in users who put male/female as their gender?
female = train.loc[train['gender'] == 'FEMALE', 'country_destination'].value_counts().sum()
male = train.loc[train['gender'] == 'MALE', 'country_destination'].value_counts().sum()

female_destinations = train.loc[train['gender'] == 'FEMALE', 'country_destination'].value_counts() / female * 100
male_destinations = train.loc[train['gender'] == 'MALE', 'country_destination'].value_counts() / male * 100

male_female_destinations_trend = pd.concat([female_destinations, male_destinations], axis=1, sort=False)
male_female_destinations_trend.columns = ['Female', 'Male']

male_female_destinations_trend_barschart = male_female_destinations_trend.plot.bar(colormap='jet',
                                                                                   title='Percentage of Gender Per Destination')
male_female_destinations_trend_barschart.set_xlabel('Country Destination')
male_female_destinations_trend_barschart.set_ylabel('Percentage')

plt.savefig('charts and diagrams/destinationspergender.png')

data = train[(train.gender == "MALE") | (train.gender == "FEMALE")]
ten_first_data = data.head(10)

# Percentage of women and men in the population study by years
years_of_study = data.date_account_created.value_counts()
gender_percentage2010 = data.gender[data.date_account_created == 2010].value_counts() / data.shape[0] * 100
gender_percentage2011 = data.gender[data.date_account_created == 2011].value_counts() / data.shape[0] * 100
gender_percentage2012 = data.gender[data.date_account_created == 2012].value_counts() / data.shape[0] * 100
gender_percentage2013 = data.gender[data.date_account_created == 2013].value_counts() / data.shape[0] * 100
gender_percentage2014 = data.gender[data.date_account_created == 2014].value_counts() / data.shape[0] * 100

gender_percentages_per_year = pd.concat([gender_percentage2010, gender_percentage2011, gender_percentage2012,
                                         gender_percentage2013, gender_percentage2014], axis=1)
gender_percentages_per_year.columns = ['2010', '2011', '2012', '2013', '2014']

gender_percentages_per_year_bar = gender_percentages_per_year.plot.bar(colormap='jet')
plt.savefig('charts and diagrams/populationbytheyears.png')

# ------------------------ Destination study by years  of account creation ------------------------
x2010_values = data.country_destination[data.date_account_created == 2010].value_counts()
x2011_values = data.country_destination[data.date_account_created == 2011].value_counts()
x2012_values = data.country_destination[data.date_account_created == 2012].value_counts()
x2013_values = data.country_destination[data.date_account_created == 2013].value_counts()
x2014_values = data.country_destination[data.date_account_created == 2014].value_counts()

destination_aggregation_by_account_creation = pd.concat(
    [x2010_values, x2011_values, x2012_values, x2013_values, x2014_values], axis=1, sort=False)
destination_aggregation_by_account_creation.columns = ['2010', '2011', '2012', '2013', '2014']

# ------------------------ Distribution by age per year ------------------------
age_info = stats.describe(data['age'])

x2010 = data[data.date_account_created == 2010]
x2011 = data[data.date_account_created == 2011]
x2012 = data[data.date_account_created == 2012]
x2013 = data[data.date_account_created == 2013]
x2014 = data[data.date_account_created == 2014]

plt.figure()
plt.boxplot([x2010.age, x2011.age, x2012.age, x2013.age, x2014.age])
plt.savefig('charts and diagrams/agesdistribution.png')

age_plot = sns.countplot(data['age'])
for ind, label in enumerate(age_plot.get_xticklabels()):
    if ind % 15 == 0:
        label.set_visible(True)
    else:
        label.set_visible(False)
plt.savefig('charts and diagrams/ageshistogram.png')

# ------------------------ Removing the outliers ------------------------

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

# ------------------------ Relationships between when the account is created and the first booking date ------------------------
clean_genderandage_data = pd.concat([new_df1, new_df2, new_df3, new_df4, new_df5])

grouped_create_date = clean_genderandage_data['date_account_created']
grouped_firstbooking_date = clean_genderandage_data['date_first_booking']
