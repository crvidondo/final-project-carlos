# IronHack Final Project 

Ironhack's Data Science & Machine Learning Bootcamp - Final Project 
Carlos Rodríguez Vidondo
8/11/24


## Project description

**Goal**: Build a predictive model to estimate the price of an Airbnb listing based on its features.

**Dataset**: 'Madrid Airbnb Data', created by Murray Cox, with information about Airbnb listings in Madrid, Spain.


## Git Repository
https://github.com/crvidondo/final-project-carlos


## Files and deliverables
The project consists of the following files:

- [datasets](https://github.com/crvidondo/final-project-carlos/tree/main/datasets): A folder containing two different .csv files: one with all datasets merged into one raw; and another that was cleaned and used later for model training. The original dataset was download from [Kaggle](https://www.kaggle.com/datasets/rusiano/madrid-airbnb-data/data?select=calendar.csv)

- [images/neighborhoods](https://github.com/crvidondo/final-project-carlos/tree/main/images/neighborhoods): Folder containing images of different Madrid's locations that are displayed in the app.

- [Tableau Dashboard](https://public.tableau.com/app/profile/carlos.rodr.guez.vidondo/viz/AirbnbFinalProject-Carlos/Dashboard1): Includes visualization charts that effectively summarize the key findings from the EDA.

- [data_merging.py](https://github.com/crvidondo/final-project-carlos/blob/main/data_merging.py): Loads multiple datasets and merges into one comprehensive DataFrame.
    
- [data_cleaning_eda.py](https://github.com/crvidondo/final-project-carlos/blob/main/data_cleaning_eda.py): Collects and analyzes data, followed by cleaning, transforming, and preprocessing steps before building the predictive model.
    
- [model_building.py](https://github.com/crvidondo/final-project-carlos/blob/main/model_building.py): A well-documented Python code that includes model selection, training, predictions performed and evaluation metrics.

- [xgboost_model.pkl](https://github.com/crvidondo/final-project-carlos/blob/main/xgboost_model.pkl): Saved model that can be loaded for making predictions.

- [scaler_carlos.pkl](https://github.com/crvidondo/final-project-carlos/blob/main/scaler_carlos.pkl): Saved scaler that was trained with the data.

- [requirements.txt](https://github.com/crvidondo/final-project-carlos/blob/main/requirements.txt): Lists all the packages required for the project.
    
- [streamlit_app.py](https://github.com/crvidondo/final-project-carlos/blob/main/streamlit_app.py): Deployment of the product into a user-friendly web app.

- [Final Project Presentation](link github): Powerpoint file that presents the project to the cohort.


## Project overview

1. **Data Merging:**
    - Multiple datasets were merged into one by using joins.
    - From the total 100 columns only 16 were essential for the project and were selected.
    - Rename columns for easy management.

2. **Data Cleaning:**
    - Rows with NaN values in 'Superhost' column were dropped. 
    - Missing values in numerical columns were handled by using imputation strategies (filling with the median).
    - Binary categorical variables (True/False) were converted to numeric formats using label encoding ('Superhost', 'Available').
    - Multi-class categorical columns ('Room Type', 'Location') were converted into numerical using One-Hot Encoding and new columns were displayed.
    - Created binary columns for specific high-value amenities that will upgrade the price for the AirBnb.

3. **Exploratory Data Analisys:**
    - Outliers in 'Price' column were removed by using the IQR method.
    - Checked distributions and visualized relationships between key features and the target variable.
    - Used correlation matrices, scatter plots and histograms to understand distributions.

3. **Machine Learning:**
    - Feature Engineering: Selected relevant numerical columns for scaling and applied StandardScaler to standardize their values.
    - Model selection: Trained multiple regression models, including Linear Regression, Random Forest, Gradient Boosting, Decision Tree, Support Vector Regressor and XGBoost.
    - Evaluation: Using metrics like RMSE, MAE, and R².
    - Hyperparameter Tuning: Applied GridSearchCV to fine-tune the hyperparameters of the Gradient Boosting model and the XGBoost, that were the two best performers.
    - Saving: best model was saved using pickle for future use.

4. **Product:**
    - Streamlit code that builds a user-friendly web application for predicting Airbnb nightly prices based on property details.


## Installation 

1. Clone the repository:
   ```bash
   git clone https://github.com/crvidondo/final-project-carlos
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the model_building.py to create the best model or simply use the XGBoost model saved into the pickle file.

4. Run the app
   ```bash
   streamlit run streamlit_app.py