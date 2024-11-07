import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open('xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the scaler
with open('scaler_carlos.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the correct feature order used during training
feature_order = [
    'Available', 'Capacity', 'Superhost', 'Bedrooms', 'Beds', 'Number of Reviews', 
    'Guest Satisfaction', 'Cleanliness Rating', 'Location Rating', 
    'Room Type_Entire home/apt', 'Room Type_Hotel room', 'Room Type_Private room', 'Room Type_Shared room',
    'Economic Class_High Class', 'Economic Class_Low', 'Economic Class_Lower-Middle', 'Economic Class_Upper-Middle',
    'Has_Pool', 'Has_Wifi', 'Has_Kitchen', 'Has_Elevator'
]

# Define the folder containing neighborhood images
image_folder = 'images/neighborhoods'

# Define only the columns that require scaling
num_cols = [
    'Available', 'Capacity', 'Superhost', 'Bedrooms', 'Beds', 'Number of Reviews', 
    'Guest Satisfaction', 'Cleanliness Rating', 'Location Rating', 
    'Room Type_Entire home/apt', 'Room Type_Hotel room', 'Room Type_Private room', 'Room Type_Shared room',
    'Economic Class_High Class', 'Economic Class_Low', 'Economic Class_Lower-Middle', 'Economic Class_Upper-Middle',
    'Has_Pool', 'Has_Wifi', 'Has_Kitchen', 'Has_Elevator'
]

# Define neighborhood-to-economic class
neighborhood_to_class = {
    'Chamartín': 'High Class', 'Latina': 'Lower-Middle', 'Arganzuela': 'Upper-Middle',
    'Centro': 'Upper-Middle', 'Salamanca': 'High Class', 'Fuencarral - El Pardo': 'Upper-Middle',
    'Puente de Vallecas': 'Low', 'Ciudad Lineal': 'Lower-Middle', 'Chamberí': 'High Class',
    'Villaverde': 'Low', 'Hortaleza': 'Upper-Middle', 'Moncloa - Aravaca': 'High Class',
    'Carabanchel': 'Low', 'Tetuán': 'Lower-Middle', 'Retiro': 'High Class',
    'San Blas - Canillejas': 'Lower-Middle', 'Villa de Vallecas': 'Low', 'Barajas': 'Lower-Middle',
    'Usera': 'Low', 'Vicálvaro': 'Low', 'Moratalaz': 'Low'
}

# Define Streamlit app
st.title("Airbnb Price Prediction App")
st.write("Input the property details to predict the nightly price.")

# Collect inputs for all features
location = st.sidebar.selectbox("Location", list(neighborhood_to_class.keys()))
capacity = st.sidebar.number_input("Capacity (number of guests)", min_value=1, max_value=20, value=2)
superhost = st.sidebar.selectbox("Superhost", ["Yes", "No"])
bedrooms = st.sidebar.number_input("Number of Bedrooms", min_value=0, max_value=20, value=1)
beds = st.sidebar.number_input("Number of beds", min_value=0, max_value=30, value=1)
num_reviews = st.sidebar.number_input("Number of Reviews", min_value=0, max_value=800, value=10)
guest_satisfaction = st.sidebar.slider("Guest Satisfaction Rating (1-100)", min_value=0, max_value=100, value=75)
cleanliness_rating = st.sidebar.slider("Cleanliness Rating (1-10)", min_value=0, max_value=10, value=8)
location_rating = st.sidebar.slider("Location Rating (1-10)", min_value=0, max_value=10, value=8)
room_type = st.sidebar.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room", "Hotel room"])
has_pool = st.sidebar.selectbox("Has Pool", ["Yes", "No"])
has_wifi = st.sidebar.selectbox("Has Wifi", ["Yes", "No"])
has_kitchen = st.sidebar.selectbox("Has Kitchen", ["Yes", "No"])
has_elevator = st.sidebar.selectbox("Has Elevator", ["Yes", "No"])

# Display neighborhood image based on location
image_path = os.path.join(image_folder, f"{location}.jpg")
if os.path.exists(image_path):
    st.image(image_path, caption=f"{location}", use_column_width=True)
else:
    st.write("Image for this location is not available.")

# Encode binary and categorical features
input_data = {
    'Available': 1,  # Assuming all entries are available
    'Capacity': capacity,
    'Superhost': 1 if superhost == "Yes" else 0,
    'Bedrooms': bedrooms,
    'Beds': beds,
    'Number of Reviews': num_reviews,
    'Guest Satisfaction': guest_satisfaction,
    'Cleanliness Rating': cleanliness_rating,
    'Location Rating': location_rating,
    'Room Type_Entire home/apt': 1 if room_type == "Entire home/apt" else 0,
    'Room Type_Hotel room': 1 if room_type == "Hotel room" else 0,
    'Room Type_Private room': 1 if room_type == "Private room" else 0,
    'Room Type_Shared room': 1 if room_type == "Shared room" else 0,
    'Has_Pool': 1 if has_pool == "Yes" else 0,
    'Has_Wifi': 1 if has_wifi == "Yes" else 0,
    'Has_Kitchen': 1 if has_kitchen == "Yes" else 0,
    'Has_Elevator': 1 if has_elevator == "Yes" else 0
}

# Encode Economic Class based on selected neighborhood
economic_class = neighborhood_to_class[location]
input_data['Economic Class_High Class'] = 1 if economic_class == 'High Class' else 0
input_data['Economic Class_Low'] = 1 if economic_class == 'Low' else 0
input_data['Economic Class_Lower-Middle'] = 1 if economic_class == 'Lower-Middle' else 0
input_data['Economic Class_Upper-Middle'] = 1 if economic_class == 'Upper-Middle' else 0

# Create DataFrame from input_data to scale the input in the same way as training
input_df = pd.DataFrame([input_data])

# Reorder columns in input_df to match the feature order in training
input_df = input_df[feature_order]

# Scale numerical columns
input_df[num_cols] = scaler.transform(input_df[num_cols])

# Predict price
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.write(f"Estimated nightly price: ${prediction:.2f}")