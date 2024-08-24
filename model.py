import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from geopy.distance import geodesic

# Load the data
data = pd.read_csv('data/train.csv')

# Data Cleaning
# Remove leading and trailing spaces, and handle missing values
data['Delivery_person_Age'] = data['Delivery_person_Age'].str.strip()
data['Delivery_person_Age'].replace('NaN', np.nan, inplace=True)
data['Delivery_person_Age'] = data['Delivery_person_Age'].astype(float).fillna(data['Delivery_person_Age'].median()).astype(int)

data['Delivery_person_Ratings'] = data['Delivery_person_Ratings'].astype(float)

# Handle 'multiple_deliveries' column similarly
data['multiple_deliveries'] = data['multiple_deliveries'].astype(str).str.strip()  # Remove any spaces
data['multiple_deliveries'].replace('NaN', np.nan, inplace=True)
data['multiple_deliveries'] = data['multiple_deliveries'].fillna(0).astype(int)

data['Time_taken(min)'] = data['Time_taken(min)'].str.extract('(\d+)').astype(int)


# Calculate distance
def calculate_distance(df):
    restaurant_coords = df[['Restaurant_latitude', 'Restaurant_longitude']].to_numpy()
    delivery_coords = df[['Delivery_location_latitude', 'Delivery_location_longitude']].to_numpy()
    df['distance'] = [geodesic(res, deliv).kilometers for res, deliv in zip(restaurant_coords, delivery_coords)]
    return df

data = calculate_distance(data)

# Drop unnecessary columns
data.drop(['ID', 'Delivery_person_ID', 'Order_Date', 'Time_Orderd', 'Time_Order_picked'], axis=1, inplace=True)

# One-hot encoding categorical variables
data = pd.get_dummies(data, columns=[
    'Weatherconditions', 'Road_traffic_density', 'Type_of_order',
    'Type_of_vehicle', 'Festival', 'City'])

# Features and target variable
x = data.drop('Time_taken(min)', axis=1)
y = data['Time_taken(min)']

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model Training
model = XGBRegressor()
model.fit(x_train, y_train)

# Model Evaluation
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"r2 score: {r2}")

# Save the model
model.save_model('delivery_time_model.json')
