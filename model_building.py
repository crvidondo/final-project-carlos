# LIBRARIES

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


# LOAD PROCESSED AND CLEANED DATA

# Load the cleaned dataset
df = pd.read_csv('../datasets/airbnb_cleaned_df.csv')

# Display the first few rows to confirm it loaded correctly
df.shape


# DEFINE FEATURE COLUMNS AND TARGET VARIABLE

X = df.drop('Price', axis=1) # features
y = df['Price'] # target


# SPLIT THE DATA INTO TRAINING AND TESTING SETS

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# FEATURE ENGINEERING

# Identify and select only numerical columns that make sense to scale
""" Columns that are one-hot encoded or that have binary values don't need to be standardized """
num_cols = ['Available', 'Capacity', 'Superhost', 'Bedrooms', 'Beds', 'Number of Reviews', 
            'Guest Satisfaction', 'Cleanliness Rating', 'Location Rating', 
            'Room Type_Entire home/apt', 'Room Type_Hotel room', 'Room Type_Private room', 'Room Type_Shared room',
            'Economic Class_High Class', 'Economic Class_Low', 'Economic Class_Lower-Middle', 'Economic Class_Upper-Middle',
            'Has_Pool', 'Has_Wifi', 'Has_Kitchen', 'Has_Elevator'
            ]

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the training set; transform the test set
X_train_scaled = scaler.fit_transform(X_train[num_cols])
X_test_scaled = scaler.transform(X_test[num_cols])

# Save the trained model into a file using pickle
with open('scaler_carlos.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Convert scaled data back to DataFrame with only the scaled columns
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=num_cols, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=num_cols, index=X_test.index)


# MODEL BUILDING AND EVALUATION

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Support Vector Regressor': SVR(),
    'XGBoost': XGBRegressor(random_state=42)
}

# Train, predict, and evaluate each model using a for loop
for name, model in models.items():
    model.fit(X_train_scaled_df, y_train)
    y_pred = model.predict(X_test_scaled_df)
    
    # Calculate evaluation metrics
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print results
    print(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")


# HYPERPARAMETER TUNING ON GRADIENT BOOSTING AND XGBOOST MODELS

# Define the parameter grid for Gradient Boosting
param_grid_gb = {
    'n_estimators': [200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [5, 7, 15],
    'min_samples_split': [10, 100, 1000]
}

# Initialize the GridSearchCV with the Gradient Boosting model
grid_search_gb = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid_gb,
    cv=3, 
    scoring='neg_mean_squared_error',
    n_jobs=-1  
)

# Fit the GridSearchCV on the training data
grid_search_gb.fit(X_train_scaled_df, y_train)

# Get the best estimator and print the best parameters
best_gb_model = grid_search_gb.best_estimator_
print("Best parameters for Gradient Boosting:", grid_search_gb.best_params_)

# Predict and evaluate using the best model
y_pred_gb = best_gb_model.predict(X_test_scaled_df)

rmse_gb = root_mean_squared_error(y_test, y_pred_gb)
mae_gb = mean_absolute_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print("Tuned Gradient Boosting - RMSE:", rmse_gb)
print("Tuned Gradient Boosting - MAE:", mae_gb)
print("Tuned Gradient Boosting - R²:", r2_gb)

# Define the parameter grid for XGBoost
param_grid_xgb = {
    'n_estimators': [500, 700, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [7, 10, 20],
    'min_child_weight': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0]
}

# Initialize the GridSearchCV with the XGBRegressor model
grid_search_xgb = GridSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_grid=param_grid_xgb,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

# Fit GridSearchCV on the training data
grid_search_xgb.fit(X_train_scaled_df, y_train)

# Get the best estimator and print the best parameters
best_xgb_model = grid_search_xgb.best_estimator_
print("Best parameters for XGBoost:", grid_search_xgb.best_params_)

# Predict and evaluate using the best model
y_pred_xgb = best_xgb_model.predict(X_test_scaled_df)

rmse_xgb = root_mean_squared_error(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print("Tuned XGBoost - RMSE:", rmse_xgb)
print("Tuned XGBoost - MAE:", mae_xgb)
print("Tuned XGBoost - R²:", r2_xgb)


# CROSS VALIDATION

cv_scores = cross_val_score(best_gb_model, X_train_scaled_df, y_train, cv=5, scoring='neg_mean_squared_error')
mean_cv_rmse = (-cv_scores.mean())**0.5
print("Gradient Boosting Cross-Validation RMSE:", mean_cv_rmse)

cv_scores = cross_val_score(best_xgb_model, X_train_scaled_df, y_train, cv=5, scoring='neg_mean_squared_error')
mean_cv_rmse = (-cv_scores.mean())**0.5
print("XGBoost Cross-Validation RMSE:", mean_cv_rmse)


# MODEL DEPLOYMENT

# Save the trained model into a file using pickle
with open('xgboost_model.pkl', 'wb') as file:
    pickle.dump(best_xgb_model, file)