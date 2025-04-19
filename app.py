import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import openai

# Load trained model
model = tf.keras.models.load_model("structure_model.h5")

# Class labels in order (from your label encoder)
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

# Set OpenAI API key from Streamlit Secrets
openai.api_key = st.secrets["openai_key"]

st.set_page_config(page_title="Omani Structure Classifier", layout="centered")
st.title("🏛️ Omani Structure Classifier + AI Heritage Assistant")
st.markdown("Upload a photo of a traditional building from Izzk or Nizwa. The model will classify it and ChatGPT will explain the architecture 🧱")

# Upload an image
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image
    st.image(uploaded_file, caption="📷 Uploaded Image", use_column_width=True)

    # Convert to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    predicted_index = np.argmax(prediction)
    predicted_label = labels[predicted_index]
    confidence = prediction[0][predicted_index]

    st.subheader(f"🏷️ Prediction: `{predicted_label}`")
    st.write(f"🔍 Confidence: **{confidence * 100:.2f}%**")

    # Ask ChatGPT for description
    with st.spinner("🧠 Asking ChatGPT for analysis..."):
        prompt = f"""
        Provide a detailed architectural and cultural analysis of the structure type: {predicted_label}.
        Include: building configuration, material, structure type, brick pattern, historical context, durability, and a text-based sketch description if possible.
        """
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        ai_description = response.choices[0].message.content
        st.markdown("### 📋 Structure Analysis")
        st.write(ai_description)

    # Follow-up Q&A
    st.markdown("### 💬 Ask a Follow-Up Question")
    user_question = st.text_input("Type your question here:")
    if user_question:
        follow_up = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in traditional Omani architecture."},
                {"role": "user", "content": user_question}
            ]
        )
        st.write("🧠", follow_up.choices[0].message.content)

st.markdown("---")
st.caption("Built by Abdullah Al-Dhahli – Sultan Qaboos University 🌍")
