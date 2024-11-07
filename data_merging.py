import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load each CSV into a DataFrame
listings_detailed_df = pd.read_csv('C:\\Users\\crvid\\Documents\\IRONHACK\\WEEK_9\\final-project-carlos\\data/listings_detailed.csv')
listings_df = pd.read_csv('C:\\Users\\crvid\\Documents\\IRONHACK\\WEEK_9\\final-project-carlos\\data/listings.csv')

# Convert 'id' columns to string in all DataFrames
listings_detailed_df['id'] = listings_detailed_df['id'].astype(str)
listings_df['id'] = listings_df['id'].astype(str)

# Merge listings with detailed listings on 'id'
merged_df = pd.merge(listings_df, listings_detailed_df, on='id', how='inner')

# Check the structure of the merged DataFrame
merged_df.head()

# Check the columns after merging and keeping the ones that are essential
merged_df.columns

# Check for some unique values
merged_df['room_type_x'].unique()

# Select only the essential columns
essential_columns = [
    'id', 'neighbourhood_group', 'price_x', 'instant_bookable', 'room_type_x', 'accommodates', 
    'host_is_superhost', 'bedrooms', 'beds', 'amenities', 'latitude_x', 'longitude_x', 
    'number_of_reviews_x', 'review_scores_rating', 'review_scores_cleanliness', 'review_scores_location'
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
    'beds': 'Beds',
    'amenities': 'Amenities',
    'latitude_x': 'Latitude',
    'longitude_x': 'Longitude',
    'number_of_reviews_x': 'Number of Reviews',
    'review_scores_rating': 'Guest Satisfaction',
    'review_scores_cleanliness': 'Cleanliness Rating',
    'review_scores_location': 'Location Rating',
}, inplace=True) 

# Display a preview of the filtered DataFrame and its shape
filtered_df.head()

filtered_df.shape

# Save it into a new .csv that its ready to be cleaned
filtered_df.to_csv('airbnb_merged_df.csv', index=False)