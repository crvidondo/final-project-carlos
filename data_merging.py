import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load each CSV into a DataFrame
calendar_df = pd.read_csv('../final-project-carlos/data/calendar.csv')
listings_detailed_df = pd.read_csv('../final-project-carlos/data/listings_detailed.csv')
listings_df = pd.read_csv('../final-project-carlos/data/listings.csv')
reviews_detailed_df = pd.read_csv('../final-project-carlos/data/reviews_detailed.csv')

# Convert 'id' columns to string in all DataFrames
calendar_df['id'] = calendar_df['id'].astype(str)
listings_detailed_df['id'] = listings_detailed_df['id'].astype(str)
listings_df['id'] = listings_df['id'].astype(str)
reviews_detailed_df['id'] = reviews_detailed_df['id'].astype(str)

# Merge listings with detailed listings on 'id'
merged_df = pd.merge(listings_df, listings_detailed_df, on='id', how='inner')
# Merge with calendar on 'id'
merged_df = pd.merge(merged_df, calendar_df, on='id', how='left')
# Merge with reviews on 'id'
merged_df = pd.merge(merged_df, reviews_detailed_df, on='id', how='left')

# Check the structure of the merged DataFrame
merged_df.head(5)

# Check the columns after merging and keeping the ones that are essential
merged_df.columns

# Check for some unique values
merged_df['room_type_x'].unique()

# Select only the essential columns
essential_columns = [
    'id', 'neighbourhood_group', 'price_x', 'instant_bookable', 'room_type_x', 'accommodates', 
    'host_is_superhost', 'bedrooms', 'bathrooms', 'amenities', 'latitude_x', 'longitude_x', 
    'number_of_reviews_x', 'review_scores_rating', 'review_scores_cleanliness'
]

# Keep only the essential columns
filtered_df = merged_df[essential_columns]

# Display a preview of the filtered DataFrame
filtered_df.head()

# Renaming columns for easy management
filtered_df.rename(columns={
    'neighbourhood_group': 'Location',
    'price_x': 'Price',
    'instant_bookable': 'Available',
    'room_type_x': 'Room Type',
    'accommodates': 'Capacity',
    'host_is_superhost': 'Superhost',
    'bedrooms': 'Bedrooms',
    'bathrooms': 'Bathrooms',
    'amenities': 'Amenities',
    'latitude_x': 'Latitude',
    'longitude_x': 'Longitude',
    'number_of_reviews_x': 'Number of Reviews',
    'review_scores_rating': 'Guest Satisfaction',
    'review_scores_cleanliness': 'Cleanliness Rating',
}, inplace=True) 

# Display a preview of the filtered DataFrame
filtered_df.head()

filtered_df.shape

# Remove duplicate rows based on all columns
filtered_df = filtered_df.drop_duplicates(subset=['id'])

filtered_df.shape

filtered_df.to_csv('airbnb_merged_df.csv', index=False)