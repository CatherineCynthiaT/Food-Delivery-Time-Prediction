import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from geopy.distance import geodesic
import base64

# Load the trained model
model = XGBRegressor()
model.load_model('delivery_time_model.json')

# CSS for background image and custom styles
page_bg_img = f'''
<style>
body {{
    background-color: #F0F8FF;  /* Alice Blue background */
    color: #2F4F4F;  /* Dark Slate Gray for the main text */
}}
header {{
    background-color: #4682B4;  /* Steel Blue for header */
    color: #FFFFFF;  /* White text color for header */
    border-radius: 12px;
    padding: 10px;
}}
.stButton button {{
    background-color: #20B2AA;  /* Light Sea Green for buttons */
    color: #FFFFFF;  /* White text color for buttons */
    border-radius: 10px;
    padding: 8px 16px;
}}
.stSlider > div > div {{
    color: #2E8B57;  /* Sea Green for slider text */
}}
.stSelectbox > div > div > div {{
    color: #2E8B57;  /* Sea Green for select box text */
}}
.stMarkdown {{
    color: #2F4F4F;  /* Dark Slate Gray for markdown text */
}}
.stNumberInput > label {{
    color: #4682B4;  /* Steel Blue for number input label */
}}
.stNumberInput > div > input {{
    background-color: #E6E6FA;  /* Lavender for input background */
    color: #2F4F4F;  /* Dark Slate Gray for input text */
}}
.stTextInput > label {{
    color: #4682B4;  /* Steel Blue for text input label */
}}
.stTextInput > div > input {{
    background-color: #E6E6FA;  /* Lavender for input background */
    color: #2F4F4F;  /* Dark Slate Gray for input text */
}}
.stRadio > div > label {{
    color: #4682B4;  /* Steel Blue for radio label */
}}
.stRadio > div > div {{
    color: #2E8B57;  /* Sea Green for radio button text */
}}
footer {{
    background-color: #4682B4;  /* Steel Blue for footer */
    color: #FFFFFF;  /* White text color for footer */
    padding: 10px;
}}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# Define the Streamlit interface
st.title("🍔 Food Delivery Time Prediction", anchor=False)

# Input fields for the user
restaurant_lat = st.number_input("Restaurant Latitude", value=0.0, format="%.6f")
restaurant_long = st.number_input("Restaurant Longitude", value=0.0, format="%.6f")
delivery_lat = st.number_input("Delivery Location Latitude", value=0.0, format="%.6f")
delivery_long = st.number_input("Delivery Location Longitude", value=0.0, format="%.6f")

age = st.slider("Delivery Person Age", 18, 60, 30)
ratings = st.slider("Delivery Person Ratings", 0.0, 5.0, 3.5)
vehicle_condition = st.selectbox("Vehicle Condition", [0, 1, 2])
multiple_deliveries = st.selectbox("Multiple Deliveries", [0, 1])

weather = st.selectbox("Weather Conditions", ['Sunny', 'Stormy', 'Sandstorms', 'Cloudy'])
traffic = st.selectbox("Road Traffic Density", ['High', 'Jam', 'Medium', 'Low'])
order_type = st.selectbox("Type of Order", ['Snack', 'Drinks', 'Meal', 'Buffet'])
vehicle_type = st.selectbox("Type of Vehicle", ['motorcycle', 'scooter'])
festival = st.selectbox("Festival", ['Yes', 'No'])
city = st.selectbox("City", ['Urban', 'Metropolitan', 'Semi-Urban'])

# Calculate distance
def calculate_distance(restaurant_lat, restaurant_long, delivery_lat, delivery_long):
    restaurant_coords = (restaurant_lat, restaurant_long)
    delivery_coords = (delivery_lat, delivery_long)
    return geodesic(restaurant_coords, delivery_coords).kilometers

distance = calculate_distance(restaurant_lat, restaurant_long, delivery_lat, delivery_long)

# Prepare input data template with zeros
input_data = pd.DataFrame(np.zeros((1, len(model.get_booster().feature_names))), columns=model.get_booster().feature_names)

# Set the values based on user input
input_data['Delivery_person_Age'] = age
input_data['Delivery_person_Ratings'] = ratings
input_data['Vehicle_condition'] = vehicle_condition
input_data['multiple_deliveries'] = multiple_deliveries
input_data['distance'] = distance

# Setting categorical features based on user selection
if f'Weatherconditions_{weather}' in input_data.columns:
    input_data[f'Weatherconditions_{weather}'] = 1
if f'Road_traffic_density_{traffic}' in input_data.columns:
    input_data[f'Road_traffic_density_{traffic}'] = 1
if f'Type_of_order_{order_type}' in input_data.columns:
    input_data[f'Type_of_order_{order_type}'] = 1
if f'Type_of_vehicle_{vehicle_type}' in input_data.columns:
    input_data[f'Type_of_vehicle_{vehicle_type}'] = 1
if f'Festival_{festival}' in input_data.columns:
    input_data[f'Festival_{festival}'] = 1
if f'City_{city}' in input_data.columns:
    input_data[f'City_{city}'] = 1

# Make predictions
if st.button("Predict Delivery Time", key="predict_button"):
    try:
        prediction = model.predict(input_data)
        st.success(f"🚚 Predicted Delivery Time: {prediction[0]:.2f} minutes")
    except ValueError as e:
        st.error(f"Prediction Error: {e}")
