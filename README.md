# Project Overview  
This project involves building a predictive model for estimating food delivery times using machine learning techniques. It focuses on forecasting delivery times based on factors such as delivery person's age, ratings, weather conditions, traffic density, multiple deliveries, and distance between the restaurant and delivery location. The project includes model development (`model.py`) and deployment via a web-based application (`app.py`).  

## Data Source  
The dataset for this project can be obtained from [Kaggle](www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset).  

## Implementation Details  

### Model Development (`model.py`)  
1. **Data Loading and Cleaning**:  
   - Processed dataset to handle missing values and clean columns such as `Delivery_person_Age` and `multiple_deliveries`.  
   - Extracted numerical values from the `Time_taken(min)` column.  

2. **Feature Engineering**:  
   - Calculated distances between restaurants and delivery locations using the `geodesic` function from the `geopy` library.  

3. **One-Hot Encoding**:  
   - Converted categorical variables (e.g., weather conditions, road traffic density, city) into numerical features for model compatibility.  

4. **Model Training**:  
   - Trained an `XGBRegressor` model to predict delivery times using training and testing splits.  
   - Evaluated model performance using metrics like MSE, RMSE, and R² score.  

5. **Model Saving**:  
   - Saved the trained model as `delivery_time_model.json` for application deployment.  

### Web-Based Application (`app.py`)  
1. **User Interface Design**:  
   - Built an interactive interface using Streamlit, allowing users to input delivery details and view predictions.  
   - Designed with custom CSS for a clean, user-friendly layout.  

2. **Prediction Mechanism**:  
   - Integrated a prediction system to process user inputs and compute delivery time estimates using the trained model.  

3. **Real-Time Prediction**:  
   - Enabled instant predictions with a "Predict Delivery Time" feature, displaying results dynamically.  

4. **Deployment**:  
   - Deployed as a standalone web application accessible via any browser.  

## Results and Evaluation  
The `XGBRegressor` model achieved an R² score of 0.82, showcasing its ability to accurately predict food delivery times. The web application provides an intuitive and practical tool for real-world applications in the food delivery sector.  
