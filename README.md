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

    - [data](Link): A folder containing different .csv files that were merged into one to be used for model training. The dataset was download from [Kaggle](https://www.kaggle.com/datasets/rusiano/madrid-airbnb-data/data?select=calendar.csv)

    - [Tableau Dashboard](link): Includes visualization charts that effectively summarize the key findings from the EDA.

    - [data_merging.py](https://github.com/crvidondo/final-project-carlos/blob/main/data_merging.py): Loads multiple datasets and merges into one comprehensive DataFrame.
    
    - [data_cleaning_eda.py](https://github.com/crvidondo/final-project-carlos/blob/main/data_cleaning_eda.py): Collects and analyzes data, followed by cleaning, transforming, and preprocessing steps before building the predictive model.
    
    - [model_building.py](https://github.com/crvidondo/final-project-carlos/blob/main/model_building.py): A well-documented Python code that includes model selection, training, predictions performed and evaluation metrics.

    - gradient_boosting_model.pkl: Saved model that can be loaded for making predictions. 
    
    - [streamlit_app.py](link): Deployment of the product into a user-friendly web app.

    - [Project presentation](link github): Powerpoint file that presents the project to the cohort.


## Project overview

1. **Data Merging:**
    - Multiple datasets were merged into one by using joins.
    - From the total 100 columns only 15 were essential for the project and were selected.
    - Duplicate rows with same 'id' were dropped.

2. **Data Cleaning:**
    - 'Bathroom' column was dropped due to lack of data.
    - Rows with NaN values in 'Superhost' column were dropped. 
    - Missing values in numerical columns were handled by using imputation strategies (filling with the median).
    - Categorical variables were converted to numeric formats using label encoding and one-hot encoding ('Superhost', 'Available', 'Room Type')

3. **Exploratory Data Analisys:**
    - Checked distributions and visualized relationships between key features and the target variable.
    - Outliers in 'Price' column were removed by using the IQR method.
    - Used correlation matrices, scatter plots and histograms to understand distributions.

3. **Machine Learning:**
    - Feature Engineering: Selected relevant numerical columns for scaling and applied StandardScaler to standardize their values.
    - Model selection: Trained multiple regression models, including Linear Regression, Random Forest, Gradient Boosting, Decision Tree, and Support Vector Regressor.
    - Evaluation: Using metrics like RMSE, MAE, and R².
    - Hyperparameter Tuning: Applied GridSearchCV to fine-tune the hyperparameters of the Gradient Boosting model, that was the bes performer.
    - Saving: model was saved using pickle for future use.

4. **Product:**
    - 


## Installation - use locally 

1. Clone the repository:
   ```bash
   git clone https://github.com/crvidondo/final-project-carlos
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app
   ```bash
   streamlit run streamlit_app.py