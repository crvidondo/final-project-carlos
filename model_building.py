# LIBRARIES

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


# LOAD PROCESSED AND CLEANED DATA

# Load the cleaned dataset
df = pd.read_csv('../airbnb_cleaned_df.csv')

# Display the first few rows to confirm it loaded correctly
df.head()


# DEFINE FEATURE COLUMNS AND TARGET VARIABLE

X = df.drop('Price', axis=1) # features
y = df['Price'] # target


# SPLIT THE DATA INTO TRAINING AND TESTING SETS

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# FEATURE ENGINEERING

# Identify and select only numerical columns that make sense to scale
""" Columns that are one-hot encoded or that have binary values don't need to be standardized """
num_cols = ['Capacity', 'Bedrooms', 'Number of Reviews', 'Guest Satisfaction', 'Cleanliness Rating']

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the training set; transform the test set
X_train_scaled = scaler.fit_transform(X_train[num_cols])
X_test_scaled = scaler.transform(X_test[num_cols])

# Convert scaled data back to DataFrame with only the scaled columns
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=num_cols, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=num_cols, index=X_test.index)

# Replace the original columns in X_train and X_test with the scaled versions
X_train.update(X_train_scaled_df)
X_test.update(X_test_scaled_df)


# MODEL BUILDING AND EVALUATION

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Support Vector Regressor': SVR(),
}

# Train, predict, and evaluate each model
for name, model in models.items():
    model.fit(X_train_scaled_df, y_train)
    y_pred = model.predict(X_test_scaled_df)
    
    # Calculate evaluation metrics
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print results
    print(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")


# HYPERPARAMETER TUNING

# Define the parameter grid for Gradient Boosting
param_grid_gb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

# Initialize the GridSearchCV with the Gradient Boosting model
grid_search_gb = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid_gb,
    cv=3,  # 3-fold cross-validation
    scoring='neg_mean_squared_error',  # Use negative MSE for scoring
    n_jobs=-1  # Use all available cores
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


# CROSS VALIDATION

cv_scores = cross_val_score(best_gb_model, X_train_scaled_df, y_train, cv=5, scoring='neg_mean_squared_error')
mean_cv_rmse = (-cv_scores.mean())**0.5
print("Cross-Validation RMSE:", mean_cv_rmse)


# MODEL DEPLOYMENT

import pickle

# Save the trained model into a file using pickle
with open('gradient_boosting_model.pkl', 'wb') as file:
    pickle.dump(best_gb_model, file)