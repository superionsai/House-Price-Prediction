import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

df = pd.read_csv('bengaluru_house_prices.csv')
df.head()
df.isnull().sum()
#We shall check if the society, area_type, balcony and availability has a relation with the price of the house
sns.scatterplot(x='society', y='price', data=df)
plt.show() #Concluded no relation between society and price

sns.scatterplot(x='area_type', y='price', data=df)
plt.show() #Super-build up area takes the mode but no proper conclusion
df['area_type'].value_counts().plot(kind='bar')
plt.show() #Conclued that the mode/mea/median for any area_type lies in the same region excluding a few outliers and different value_counts
#Hence society and area_type can be removed from the dataset

sns.scatterplot(x='balcony', y='price', data=df)
plt.show()

df['availability'].value_counts().plot(kind='bar')
plt.show() #Concluded that availability has no relation with the price of the house
#Most of the houses are ready to move so doesn't matter
df = df.drop(['area_type', 'society', 'balcony', 'availability'], axis=1)
df.isnull().sum() #Checking for null values again
df['size'].dropna(inplace=True) #Dropping null values in size
df['size'] = df['size'].astype(str) #Converting size to string
#Checking for all unique values and presence of wrong formatted values
for col in df.columns:
    print(col, df[col].unique(), df[col].nunique())
    print(df[col].dtype)
#size has many repeat values and some values in total sqft are given in range
for item in df['size']: # Loop through each item in the 'size' column
    df['size'] = df['size'].replace(item, item.split(' ')[0])
def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
df['total_sqft'].dtypes
def convert_to_float(value):
    if isinstance(value, str):
        if '-' in value:  # Check for range values
            parts = value.split('-')
            return (float(parts[0]) + float(parts[1])) / 2  # Return the average of the range
        elif is_float(value):  # Check if it's a float
            return float(value)
    return np.nan  # Return NaN for non-convertible values
df['total_sqft'] = df['total_sqft'].apply(convert_to_float)  # Apply the conversion function to the 'total_sqft' column
df['total_sqft'] = df['total_sqft'].astype(float)  # Convert the column to float type
df.head() #Checking the head of the dataframe after cleaning
df['price_per_sqft'] = df['price']*100000 / df['total_sqft']  # Creating a new column for price per square foot
df.head()
df['location'].value_counts() #checking briefly the range of locations count
local = df.groupby('location')['location'].agg('count').sort_values(ascending = False)
df['location'] = df['location'].apply(lambda x: 'other' if x in local[local<=10] else x)
len(df['location'].unique()) #remove too many categories thereby cleaning random locations
#To check outliers in price/sqft as there can be some data that is wrong
#Practical prices will only be taken
sns.boxplot(x=df['price_per_sqft'])
Q1 = df['price_per_sqft'].quantile(0.25)
Q3 = df['price_per_sqft'].quantile(0.75)
IQR = Q3 - Q1
#Removing all outliers in the dataset of price_per_sqft
lower_bound= Q1 - 1.5*IQR
upper_bound= Q3 + 1.5*IQR
df_mod = df[(df['price_per_sqft'] > lower_bound) & (df['price_per_sqft'] < upper_bound)]
sns.boxplot(x=df_mod['price_per_sqft'])
df_mod['price_per_sqft'].describe()
#We are considering minimum price per sqft to be 1000 Rupees
df_mod = df_mod[df_mod['price_per_sqft']>1000]
#checking unique values
df_mod = df_mod[df_mod['size'] != 'nan']
df_mod['size'] =df_mod['size'].apply(lambda x: int(x))
df_mod['size'].unique() 

#2 BHK flats in every area are supposed to be cheaper than 3 BHK
#We have to remove such outliers also
#Assuming like this and removing may remove necessary data so what we can do is
""" We can see the sqft per BHK and if it is less than 150 sqft per bedroom, like taking washroom and
hall balcony considered as minimum of 200 sqft and bhk*150"""
df_mod = df_mod[(df_mod['total_sqft'] >= 200 + (df_mod['size'] * 140)) & (df_mod['total_sqft'] <= 1500 + df_mod['size'] * 1200)]
df_mod.shape
plt.plot(df['price_per_sqft'])
plt.xlabel('Price per Sqaure feet')
plt.ylabel('Count')
df_mod.shape #check the shape
df_mod.head()
columns = ['location', 'size', 'total_sqft', 'bath']
for column in columns:
    sns.scatterplot(x=df[column], y=df_mod['price_per_sqft'])
    plt.show()
#Check the plot for all features
#NOTE: Not much relation between location and prices have been observed, likely that the locations are nearby or of the same tier
#We can ignore the location column for finding the model
d=df_mod.drop(columns='location')
d['bath'].fillna(d['bath'].mode()[0], inplace = True)
X=d.iloc[:, :-1]
y=d.iloc[:, -1]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.9, random_state = 7)
"""Checked for test size 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, and all of them using random forest were result in a good accuracy score
of around 1 for train and 0.8 for test, the data is probably close and in a structured manner, hence the result
There is very slight but considerable overfitting therefore we can proceed with this model, this concludes my first project"""
df_mod.isnull().sum()
df_mod['bath'].fillna(df_mod['bath'].mode()[0], inplace=True)
df_mod.dropna()
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, Y_train)
lr.score(X_train, Y_train), lr.score(X_test, Y_test)
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, Y_train)
knn.score(X_train, Y_train), lr.score(X_test, Y_test)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=10,random_state=42)
rf.fit(X_train, Y_train)
rf.score(X_train, Y_train), lr.score(X_test, Y_test)
from sklearn.metrics import mean_squared_error, r2_score
Y_pred = rf.predict(X_test)
print("MSE:", mean_squared_error(Y_test, Y_pred))
print("R2 Score:", r2_score(Y_test, Y_pred))

