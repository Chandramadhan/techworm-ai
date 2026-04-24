import os
import time
import gdown
import streamlit as st
import numpy as np
import warnings
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from langchain_groq import ChatGroq
from langdetect import detect
from deep_translator import GoogleTranslator
from dotenv import load_dotenv, find_dotenv

# Silence warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*PydanticDeprecatedSince20.*")

load_dotenv('.env', override=True)

# Load LLM
def load_llm():
    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

# Translation utils
def translate_to_english(text):
    if text.lower().strip() in ["hi", "hello", "hey", "howdy"]:
        return text, "en"
        
    try:
        lang = detect(text)
        if len(text.split()) <= 2 and lang != "en":
             return text, "en"
        return text if lang == "en" else GoogleTranslator(source=lang, target="en").translate(text), lang
    except:
        return text, "unknown"

def translate_back(text, lang):
    return text if lang == "en" else GoogleTranslator(source="en", target=lang).translate(text)

# Download + load model from Drive
@st.cache_resource
def load_disease_model():
    model_path = "plant_disease_model.h5"
    if not os.path.exists(model_path):
        with st.spinner("⬇️ Downloading model..."):
            gdown.download(id="19yZ3C23HzMaHmk8Ahb4ovyZ0Px37_vX9", output=model_path, quiet=False)
    return load_model(model_path, compile=False)

model = load_disease_model()

# Label map
label_map = {
    "Apple___Apple_scab": 0, "Apple___Black_rot": 1, "Apple___Cedar_apple_rust": 2, "Apple___healthy": 3,
    "Blueberry___healthy": 4, "Cherry_(including_sour)___Powdery_mildew": 5, "Cherry_(including_sour)___healthy": 6,
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": 7, "Corn_(maize)___Common_rust_": 8,
    "Corn_(maize)___Northern_Leaf_Blight": 9, "Corn_(maize)___healthy": 10,
    "Grape___Black_rot": 11, "Grape___Esca_(Black_Measles)": 12, "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": 13,
    "Grape___healthy": 14, "Orange___Haunglongbing_(Citrus_greening)": 15, "Peach___Bacterial_spot": 16,
    "Peach___healthy": 17, "Pepper,_bell___Bacterial_spot": 18, "Pepper,_bell___healthy": 19,
    "Potato___Early_blight": 20, "Potato___Late_blight": 21, "Potato___healthy": 22,
    "Raspberry___healthy": 23, "Soybean___healthy": 24, "Squash___Powdery_mildew": 25,
    "Strawberry___Leaf_scorch": 26, "Strawberry___healthy": 27, "Tomato___Bacterial_spot": 28,
    "Tomato___Early_blight": 29, "Tomato___Late_blight": 30, "Tomato___Leaf_Mold": 31,
    "Tomato___Septoria_leaf_spot": 32, "Tomato___Spider_mites Two-spotted_spider_mite": 33,
    "Tomato___Target_Spot": 34, "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 35,
    "Tomato___Tomato_mosaic_virus": 36, "Tomato___healthy": 37
}
inv_label_map = {v: k for k, v in label_map.items()}

def predict_disease(img):
    try:
        # 1. Match ACTUAL model input shape (224x224)
        img_size = 224
        img_resized = img.resize((img_size, img_size))
        img_array = keras_image.img_to_array(img_resized)
        
        # 2. Advanced Validation (Color & Texture)
        hsv_img = img_resized.convert('HSV')
        h, s, v = hsv_img.split()
        if np.mean(s) < 30 or np.std(img_array) < 25:
            return "❌ Specimen Rejected: Image lacks the color depth or texture of a leaf. Please use a clearer, well-lit photo."

        # 3. Test-Time Augmentation (TTA)
        img_orig = np.expand_dims(img_array, axis=0).astype('float32')
        img_flip = np.expand_dims(np.fliplr(img_array), axis=0).astype('float32')
        img_rot = np.expand_dims(np.rot90(img_array), axis=0).astype('float32')
        
        preds = []
        for i in [img_orig, img_flip, img_rot]:
            preds.append(model.predict(i, verbose=0)[0])
        
        avg_pred = np.mean(preds, axis=0)
        confidence = float(np.max(avg_pred))
        class_index = int(np.argmax(avg_pred))
        result = inv_label_map.get(class_index, 'Unknown')
        
        if confidence > 0.98: confidence = 0.975
        
        if confidence < 0.60:
            return f"🔬 Uncertain Result ({confidence*100:.1f}%): Possibly {result}. Suggest consulting the AI Expert for a manual symptoms check."
            
        return f"Prediction: {result} ({confidence*100:.1f}%)"
    except Exception as e:
        return f"Error during prediction: {str(e)}"

# Streamlit UI Redesign
def main():
    st.set_page_config(page_title="Techworm AI", page_icon="🌾", layout="wide")
    
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Poppins:wght@700&display=swap');
        
        .stApp {
            background: linear-gradient(rgba(0, 20, 10, 0.85), rgba(0, 20, 10, 0.85)), 
                        url("https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1600&q=80");
            background-size: cover;
            font-family: 'Inter', sans-serif;
            color: white !important;
        }

        .main-title {
            font-family: 'Poppins', sans-serif;
            font-size: 4rem !important;
            font-weight: 800;
            background: linear-gradient(90deg, #00ff88, #60efff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0px;
        }
        
        .sub-title {
            color: #a0aec0;
            text-align: center;
            font-size: 1.3rem;
            margin-bottom: 50px;
            letter-spacing: 2px;
        }

        .stChatMessage {
            background: rgba(255, 255, 255, 0.05) !important;
            border-radius: 15px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            margin-bottom: 10px;
        }

        .stAlert {
            background: rgba(0, 255, 136, 0.1) !important;
            color: #00ff88 !important;
            border: 1px solid #00ff88 !important;
            border-radius: 15px !important;
        }
        
        h1, h2, h3, p, span, label, .stMarkdown {
            color: white !important;
        }

        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 30px;
        }
        </style>
        
        <h1 class='main-title'>TECHWORM AI</h1>
        <p class='sub-title'>SMART AGRICULTURE & PLANT INTELLIGENCE</p>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 📷 Plant Health Diagnostic")
        
        with st.expander("💡 Tips for 100% Accuracy"):
            st.write("- Use bright, natural light.\n- Keep the leaf flat and centered.\n- Ensure the focus is sharp on the spots.")

        uploaded_image = st.file_uploader("Drop a leaf photo here...", type=["jpg", "jpeg", "png"])
        
        if uploaded_image:
            img = Image.open(uploaded_image)
            st.image(img, use_container_width=True)
            with st.spinner("🔬 Analyzing specimen..."):
                result = predict_disease(img)
                st.success(f"#### {result}")
                st.session_state.last_scan = result
                st.info("🤖 **AI Note:** Shared with expert. Ask: 'Is this prediction correct?'")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 💬 Agriculture Expert")
        
        if st.button("Clear History"):
            st.session_state.messages = []
            if 'last_scan' in st.session_state: del st.session_state.last_scan
            st.rerun()

        chat_container = st.container(height=400)
        with chat_container:
            for message in st.session_state.messages:
                st.chat_message(message["role"]).markdown(message["content"])

        prompt = st.chat_input("Ask any farming question...")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").markdown(prompt)

            try:
                translated, lang = translate_to_english(prompt)
                st.caption(f"🌐 {lang.upper()} DETECTED")
                
                if lang != "en":
                    st.info(f"**Translation:** {translated}")

                scan_context = f"\n(Note: User just scanned a leaf and model predicted: {st.session_state.get('last_scan', 'No scan yet')})"
                
                llm = load_llm()
                final_prompt = f"""
You are an accurate Senior Agronomist. Provide precise farming advice.
User's Question: {translated}
{scan_context}
"""
                if translated.lower().strip() in ["hi", "hello", "hey", "howdy"]:
                    output = "Hello! I am your AI Agriculture Expert. How can I help you today?"
                else:
                    with st.spinner("💡 Thinking..."):
                        response = llm.invoke(final_prompt)
                        output = response.content

                final_response = translate_back(output, lang)
                st.chat_message("assistant").markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
