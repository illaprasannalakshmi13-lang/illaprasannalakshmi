import streamlit as st
import pickle

# Page config
st.set_page_config(page_title="AI Phishing Detector", page_icon="🤖", layout="centered")

# Custom CSS Styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

.big-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #00ffff;
}

.sub-text {
    text-align: center;
    font-size: 18px;
    color: #cbd5e1;
}

.result-safe {
    color: #00ff88;
    font-size: 24px;
    font-weight: bold;
}

.result-phish {
    color: #ff4b4b;
    font-size: 24px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Load model
model, vectorizer = pickle.load(open("model.pkl", "rb"))

# AI Robot Image
st.image("https://cdn-icons-png.flaticon.com/512/4712/4712027.png", width=120)

# Title
st.markdown('<div class="big-title">🤖 AI Based Phishing Email Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Analyze emails using Machine Learning & NLP</div>', unsafe_allow_html=True)

st.write("")

# Input Box
email = st.text_area("📩 Enter Email Content Here", height=150)

# Analyze Button
if st.button("🚀 Analyze Email"):

    X = vectorizer.transform([email])
    prediction = model.predict(X)
    probability = model.predict_proba(X)

    confidence = round(max(probability[0]) * 100, 2)

    if prediction[0] == 1:
        st.markdown('<div class="result-phish">⚠️ Phishing Email Detected</div>', unsafe_allow_html=True)
        st.write(f"🔎 Confidence: {confidence}%")
    else:
        st.markdown('<div class="result-safe">✅ Safe Email</div>', unsafe_allow_html=True)
        st.write(f"🔎 Confidence: {confidence}%")

# Sidebar
st.sidebar.title("🛡 About This Project")
st.sidebar.write("""
This AI system detects phishing emails using:

✔ Natural Language Processing  
✔ Logistic Regression Model  
✔ Real-world dataset  

Developed for Hackathon 🚀
""")