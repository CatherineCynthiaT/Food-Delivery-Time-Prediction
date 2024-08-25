### Project Overview

This project focuses on creating a predictive model for estimating food delivery times using machine learning techniques. The primary goal is to forecast how long it will take for a food order to reach a customer based on various factors such as the delivery person's age, ratings, multiple deliveries, weather conditions, traffic density, and the distance between the restaurant and the delivery location. This project is divided into two main components: model development (`model.py`) and deployment through a web-based application (`app.py`).

### Data Source

The dataset used for this project can be obtained from here. https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset

### Implementation Details

**Model Development (`model.py`):**

1. **Data Loading and Cleaning:**
The dataset is loaded and subjected to rigorous cleaning, including removing spaces and handling missing values. For instance, the 'Delivery_person_Age' and 'multiple_deliveries' columns are cleaned and converted to appropriate numeric types. The 'Time_taken(min)' column is also processed to extract numerical values for accurate predictions.

2. **Feature Engineering:**
A critical aspect of the model involves calculating the distance between the restaurant and the delivery location using geographic coordinates. This distance is computed using the `geodesic` function from the `geopy` library and added as a new feature in the dataset.

3. **One-Hot Encoding:**
Categorical variables such as weather conditions, road traffic density, type of order, type of vehicle, and city are transformed into numerical format using one-hot encoding. This step ensures that these variables can be effectively utilized by the machine learning model.

4. **Model Training:**
The cleaned and processed dataset is split into training and testing sets. The `XGBRegressor` model from the XGBoost library is then trained on the training data to predict delivery times. The model's performance is evaluated using metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score.

5. **Model Saving:**
After training, the model is saved as `delivery_time_model.json` for later use in the application.

**Web-Based Application (`app.py`):**

1. **User Interface Design:**
The web application is developed using Streamlit, providing an interactive interface where users can input various details such as geographic coordinates, delivery person's age, ratings, and more. The app also includes a custom CSS design to enhance the user experience with a clean, visually appealing layout.

2. **Prediction Mechanism:**
The application allows users to input the required features, including calculating the distance between the restaurant and the delivery location. The user inputs are then used to create a data frame that mirrors the model's expected input format.

3. **Real-Time Prediction:**
Upon clicking the "Predict Delivery Time" button, the application uses the pre-trained XGBoost model to predict the estimated delivery time. The result is displayed on the screen, providing an intuitive and immediate response to the user's input.

4. **Deployment:**
The application is deployed as a standalone web interface, allowing users to access the delivery time prediction tool from any browser.

### Results and Evaluation

The XGBoost model demonstrated robust performance with an R² score of 0.82, indicating a strong correlation between the predicted and actual delivery times. The web application allows users to interactively predict delivery times, making the model's functionality accessible and practical for real-world applications in the food delivery industry.
