import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import openai

# Load the trained model
model = tf.keras.models.load_model("structure_model.h5")

# List of class labels
labels = [
    "izzk_building_fort_or_burj",
    "izzk_building_house",
    "izzk_building_market_or_house_1_or_2_floor",
    "izzk_building_outer_colony_wall",
    "nizwa_building_house_wall_2_floor",
    "nizwa_building_house_with_round_bricks",
    "nizwa_building_mixed_brick_pattern",
    "nizwa_building_unknown_structure"
]

# OpenAI API key
openai.api_key = st.secrets["openai_key"]

# App layout
st.title("🏛️ Structure Classifier & AI Heritage Assistant")
st.write("Upload an image of a traditional Omani structure, and get an intelligent architectural analysis.")

# File uploader
uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)
    predicted_index = np.argmax(prediction)
    predicted_label = labels[predicted_index]
    confidence = prediction[0][predicted_index]

    st.subheader(f"🏷️ Prediction: `{predicted_label}` ({confidence * 100:.2f}% confidence)")

    # Ask ChatGPT for structure details
    with st.spinner("🔍 Asking the AI expert..."):
        prompt = f"""
        Provide a detailed architectural and cultural analysis of the structure type: {predicted_label}.
        Include: building configuration, material, structure type, brick pattern, historical context, durability and a text-based sketch description if possible.
        """
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        ai_description = response.choices[0].message.content
        st.write(ai_description)

    # Ask follow-up questions
    st.subheader("💬 Ask a follow-up question:")
    user_question = st.text_input("Enter your question about this structure")
    if user_question:
        follow_up = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in Omani architecture."},
                {"role": "user", "content": user_question}
            ]
        )
        st.write(f"🧠 {follow_up.choices[0].message.content}")
