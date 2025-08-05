import streamlit as st
import pickle

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit UI
st.title("📩 Spam Message Classifier")
st.write("Enter a message below and the model will predict if it's **Spam** or **Ham (Not Spam)**.")

user_input = st.text_area("Enter your message here:", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        user_transformed = vectorizer.transform([user_input])
        prediction = model.predict(user_transformed)[0]
        if prediction == 1:
            st.error("🚫 This message is classified as **SPAM**.")
        else:
            st.success("✅ This message is classified as **HAM (Not Spam)**.")
