# Import Libraries
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the pre-trained model and encoders
model_path = 'models/app_success_model.keras'
model = tf.keras.models.load_model(model_path)

# Load the encoders from the saved files
category_encoder = joblib.load('models/category_encoder.joblib')
app_type_encoder = joblib.load('models/app_type_encoder.joblib')
content_rating_encoder = joblib.load('models/content_rating_encoder.joblib')
size_category_encoder = joblib.load('models/size_category_encoder.joblib')

# Load the target encoder used during training (for decoding predictions)
target_encoder = joblib.load('models/target_encoder.joblib')
# Load the scaler used during training
scaler = joblib.load('models/scaler.joblib')

# Streamlit app configuration
st.set_page_config(page_title="Mobile App Success Predictor", page_icon="📱", layout="wide")

# Adding an image at the top
image_path = "Banner/app_banner.png"
image = Image.open(image_path)

# Resize the image to recommended size
new_width = 800
new_height = 200
resized_image = image.resize((new_width, new_height))

# Display the resized image
st.image(resized_image, use_column_width=True)

# st.header("Mobile App Success Predictor")
st.write("""
This app predicts the likelihood of success for a mobile app based on key characteristics. Insights from the prediction can guide 
developers on potential success levels, classified **Low**, **Moderate**, or **High**.
""")

# User Inputs
category = st.selectbox("Select the app category:",
                        ['ART_AND_DESIGN', 'AUTO_AND_VEHICLES', 'BEAUTY', 'BOOKS_AND_REFERENCE',
                         'BUSINESS', 'COMICS', 'COMMUNICATION', 'DATING', 'EDUCATION', 'ENTERTAINMENT',
                         'EVENTS', 'FINANCE', 'FOOD_AND_DRINK', 'HEALTH_AND_FITNESS', 'HOUSE_AND_HOME',
                         'LIBRARIES_AND_DEMO', 'LIFESTYLE', 'GAME', 'FAMILY', 'MEDICAL', 'SOCIAL',
                         'SHOPPING', 'PHOTOGRAPHY', 'SPORTS', 'TRAVEL_AND_LOCAL', 'TOOLS',
                         'PERSONALIZATION', 'PRODUCTIVITY', 'PARENTING', 'WEATHER', 'VIDEO_PLAYERS',
                         'NEWS_AND_MAGAZINES', 'MAPS_AND_NAVIGATION'])

size_mb = st.number_input("Enter the app size in MB:", min_value=0.0, step=1.0)


# Function to categorize the size
def categorize_size(size_in_mb):
    if size_in_mb <= 10:
        return "Small"
    elif 10 < size_in_mb <= 50:
        return "Medium"
    elif size_in_mb > 100:
        return "Large"
    else:
        return "Varies"


size_category = categorize_size(size_mb)

audience = st.selectbox("Select the target audience content rating:",
                        ['Everyone', 'Teen', 'Everyone 10+', 'Mature 17+', 'Adults only 18+', 'Unrated'])

license_type = st.selectbox("Is the app free or paid?", ['Free', 'Paid'])

if license_type == 'Paid':
    price = st.number_input("Enter the price of the app:", min_value=0.0, step=1.0)
else:
    price = 0.0

# Define Android version options
android_versions = {
    4.0: "4.0", 4.2: "4.2", 4.4: "4.4", 2.3: "2.3", 3.0: "3.0", 4.1: "4.1",
    0.0: "Varies with device", 2.2: "2.2", 5.0: "5.0", 6.0: "6.0", 1.6: "1.6",
    1.5: "1.5", 2.1: "2.1", 7.0: "7.0", 5.1: "5.1", 4.3: "4.3", 2.0: "2.0",
    3.2: "3.2", 7.1: "7.1", 8.0: "8.0", 3.1: "3.1", 1.0: "1.0"
}

android_version = st.selectbox("Minimum Android Version:", options=list(android_versions.values()))

# Map back to numerical value for prediction
android_version_input = [k for k, v in android_versions.items() if v == android_version][0]

# Manually map categories to integers
category_mapping = {
    'ART_AND_DESIGN': 0, 'AUTO_AND_VEHICLES': 1, 'BEAUTY': 2, 'BOOKS_AND_REFERENCE': 3,
    'BUSINESS': 4, 'COMICS': 5, 'COMMUNICATION': 6, 'DATING': 7, 'EDUCATION': 8, 'ENTERTAINMENT': 9,
    'EVENTS': 10, 'FINANCE': 11, 'FOOD_AND_DRINK': 12, 'HEALTH_AND_FITNESS': 13, 'HOUSE_AND_HOME': 14,
    'LIBRARIES_AND_DEMO': 15, 'LIFESTYLE': 16, 'GAME': 17, 'FAMILY': 18, 'MEDICAL': 19, 'SOCIAL': 20,
    'SHOPPING': 21, 'PHOTOGRAPHY': 22, 'SPORTS': 23, 'TRAVEL_AND_LOCAL': 24, 'TOOLS': 25,
    'PERSONALIZATION': 26, 'PRODUCTIVITY': 27, 'PARENTING': 28, 'WEATHER': 29, 'VIDEO_PLAYERS': 30,
    'NEWS_AND_MAGAZINES': 31, 'MAPS_AND_NAVIGATION': 32
}

# Map the selected category to its corresponding integer
category_int = category_mapping.get(category)

# Map size category to integer
size_category_mapping = {
    'Small': 0,
    'Medium': 1,
    'Large': 2,
    'Varies': 3
}

# Map the selected size category to its corresponding integer
size_category_int = size_category_mapping.get(size_category)

# Manually map audience content ratings to integers
content_rating_mapping = {
    'Everyone': 0,
    'Teen': 1,
    'Everyone 10+': 2,
    'Mature 17+': 3,
    'Adults only 18+': 4,
    'Unrated': 5
}

# Map the selected audience to its corresponding integer
content_rating_int = content_rating_mapping.get(audience)

# Map license type to integer (app_type)
app_type_mapping = {
    'Free': 0,
    'Paid': 1
}

# Map the selected license type to its corresponding integer
app_type_int = app_type_mapping.get(license_type)

# Prepare input array for the model
# Ensure the order of inputs matches the features used during model training
input_values = np.array([[category_int, size_category_int, content_rating_int, app_type_int, price, android_version_input]])

# Scale the input features
input_values_scaled = scaler.transform(input_values)

low_range = [0, 1000]
moderate_range = [1000, 1000000]
high_range = [1000000, float('inf')]

# Prediction logic
if st.button("Predict"):
    try:
        # Make predictions using the pre-trained model
        prediction = model.predict(input_values_scaled)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class = target_encoder.inverse_transform([predicted_class_index])[0]
        prediction_probabilities = prediction[0]

        # Display the predicted category
        st.write(f"Predicted Success Category: **{predicted_class}**")

        # Provide explanations based on predicted category and install ranges
        if predicted_class == "Low":
            st.write(f"""
            **Low** success category means the app is likely to have a low chance of success, with an estimated install count range of **{low_range[0]} to {low_range[1]} installs**.
            This might be due to factors such as:
            - The app's target audience or category may not be widely appealing.
            - The app's size or pricing may not be ideal for the intended audience.
            - The compatibility with Android versions might limit its user base.
            - Additionally, the paid app pricing could impact user acquisition.
            """)

        elif predicted_class == "Moderate":
            st.write(f"""
            **Moderate chance of success** means the app has a moderate likelihood of doing well, with an estimated install count range of **{moderate_range[0]} to {moderate_range[1]} installs**.
            The app could improve its chances with some optimizations, such as:
            - Better targeting of audience or adjusting its pricing model.
            - Expanding compatibility to more Android versions.
            """)

        elif predicted_class == "High":
            st.write(f"""
            **High** success category means the app is likely to do very well, with an estimated install count of **{high_range[0]} installs and above**.
            Factors contributing to this might include:
            - A popular category with broad audience appeal.
            - Competitive pricing or free availability.
            - Compatibility with a wide range of Android devices.
            """)

        else:
            st.write("""
            **Unknown**: The app's success category could not be determined. Please check your input data.
            """)


    except Exception as e:
        st.write(f"Error: {e}")
