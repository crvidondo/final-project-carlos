# IMPORT NECESSARY LIBRARIES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# 1. LOAD THE DATASET AND EXAMINE THE STRUCTURE

# Load the dataset
df = pd.read_csv('C:\\Users\\crvid\\Documents\\IRONHACK\\WEEK_9\\final-project-carlos\\datasets\\airbnb_merged_df.csv')

# Display the first few rows
df.head()

# Display the structure of the DataFrame
df.info()

df.describe()


# 2. HANDLE MISSING DATA

# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Drop rows with missing values in the 'Superhost' column
df.dropna(subset=['Superhost'], inplace=True)

# Fill missing values for numerical columns with the median
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Verify that there are no missing values left
print("Missing values after handling:")
print(df.isnull().sum())


# 3. HANDLE CATEGORICAL VARIABLES

# Encode binary categorical variables (True/False) using Label Encoding
df['Available'] = df['Available'].apply(lambda x: 1 if x == 't' else 0)
df['Superhost'] = df['Superhost'].apply(lambda x: 1 if x == 't' else 0)

# One-hot encode multi-class categorical variables
df = pd.get_dummies(df, columns=['Room Type'], drop_first=False)

# Changing bool columns to int
df['Room Type_Entire home/apt'] = df['Room Type_Entire home/apt'].astype(int)
df['Room Type_Hotel room'] = df['Room Type_Hotel room'].astype(int)
df['Room Type_Private room'] = df['Room Type_Private room'].astype(int)
df['Room Type_Shared room'] = df['Room Type_Shared room'].astype(int)

# Convert the 21 unique locations (or neighborhoods) into four economic classes

# Dictionary to map neighborhoods to economic classes
neighborhood_to_class = {
    'Arganzuela': 'Upper-Middle',
    'Barajas': 'Lower-Middle',
    'Carabanchel': 'Low',
    'Centro': 'Upper-Middle',
    'Chamartín': 'High Class',
    'Chamberí': 'High Class',
    'Ciudad Lineal': 'Lower-Middle',
    'Fuencarral - El Pardo': 'Upper-Middle',
    'Hortaleza': 'Upper-Middle',
    'Latina': 'Lower-Middle',
    'Moncloa - Aravaca': 'High Class',
    'Moratalaz': 'Low',
    'Puente de Vallecas': 'Low',
    'Retiro': 'High Class',
    'Salamanca': 'High Class',
    'San Blas - Canillejas': 'Lower-Middle',
    'Tetuán': 'Lower-Middle',
    'Usera': 'Low',
    'Vicálvaro': 'Low',
    'Villa de Vallecas': 'Low',
    'Villaverde': 'Low',
}

# Create a new column 'Economic Class' based on the neighborhood
df['Economic Class'] = df['Location'].map(neighborhood_to_class)

# One-hot encode multi-class categorical variables
df = pd.get_dummies(df, columns=['Economic Class'], drop_first=False)

# Changing bool columns to int
df['Economic Class_High Class'] = df['Economic Class_High Class'].astype(int)
df['Economic Class_Low'] = df['Economic Class_Low'].astype(int)
df['Economic Class_Lower-Middle'] = df['Economic Class_Lower-Middle'].astype(int)
df['Economic Class_Upper-Middle'] = df['Economic Class_Upper-Middle'].astype(int)

# Create binary columns for specific high-value amenities that will upgrade the price for the AirBnb
df['Has_Pool'] = df['Amenities'].apply(lambda x: 1 if 'Pool' in x else 0)
df['Has_Wifi'] = df['Amenities'].apply(lambda x: 1 if 'Wifi' in x else 0)
df['Has_Kitchen'] = df['Amenities'].apply(lambda x: 1 if 'Kitchen' in x else 0)
df['Has_Elevator'] = df['Amenities'].apply(lambda x: 1 if 'Elevator' in x else 0)


# 4. OUTLIER DETECTION

# Boxplot for detecting outliers in the 'price' column
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Price'])
plt.title('Boxplot for Price')
plt.show()

# Remove outliers using the IQR method
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Price'] >= (Q1 - 1.5 * IQR)) & (df['Price'] <= (Q3 + 1.5 * IQR))]

# Visualize the boxplot again after removing outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Price'])
plt.title('Boxplot for Price after removing outliers')
plt.show()


# 5. VISUALIZE RELATIONSHIPS

# Scatter plot of Capacity vs. Price
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Capacity'], y=df['Price'])
plt.title('Scatter Plot of Accommodates vs. Price')
plt.xlabel('Number of Accommodates')
plt.ylabel('Price')
plt.show()


# 6. CORRELATION MATRIX

# Select only numerical columns for correlation
numerical_columns = ['Price', 'Capacity', 'Bedrooms', 'Number of Reviews', 'Guest Satisfaction', 'Cleanliness Rating']
numerical_df = df[numerical_columns]

# Heatmap to visualize correlations
plt.figure(figsize=(10, 6))
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# 7. CHECK DATA DISTRIBUTION

# Histogram for price distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Price'], bins=50, kde=True)
plt.title('Price Distribution')
plt.xlabel('Price per night')
plt.show()


# 8. SAVE THE CLEANED DATASET

# Save the cleaned and prepared dataset to a CSV file
df.to_csv('airbnb_cleaned_df.csv', index=False)