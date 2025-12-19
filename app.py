# 🩺 APP STREAMLIT - VERSION PROFESSIONNELLE
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings
from fpdf import FPDF

# 🧹 Cache + warnings
st.cache_data.clear()
st.cache_resource.clear()
warnings.filterwarnings("ignore", category=UserWarning)

# 📊 Configuration page
st.set_page_config(
    page_title="DiabCheck Pro | IA de diagnostic",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 CSS professionnel
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Segoe UI', sans-serif;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        text-align: center;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 0.75rem;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(238, 90, 82, 0.4);
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ff6b6b, #5f27cd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# 🧠 Session State : initialise les variables
if "proba" not in st.session_state:
    st.session_state.proba = 0.0
if "pred" not in st.session_state:
    st.session_state.pred = 0
if "analyse_done" not in st.session_state:
    st.session_state.analyse_done = False

# 📦 Chargement modèles
@st.cache_resource
def load_models():
    with open('models/diabetes_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_models()

# 🧭 Header
st.markdown('<div class="main-title">DiabCheck Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">IA de diagnostic du diabète | Précision 95 %</div>', unsafe_allow_html=True)

# 📊 Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🧬 Données patient")
    age = st.slider("Âge", 18, 100, 30, help="Âge chronologique")
    gender = st.radio("Genre", ["Masculin", "Féminin"], horizontal=True, help="Sexe biologique")
    bmi = st.slider("IMC (kg/m²)", 10.0, 50.0, 25.0, 0.1, help="Indice de Masse Corporelle")

with col2:
    st.markdown("### ⚕️ Symptômes cliniques")
    polyuria = st.checkbox("🚰 Polyurie (urination fréquente)")
    polydipsia = st.checkbox("🥤 Polydipsie (soif excessive)")
    weight_loss = st.checkbox("⚖️ Perte de poids soudaine")
    paresis = st.checkbox("🖐️ Paresthésie (engourdissement)")
    polyphagia = st.checkbox("🍽️ Polyphagie (faim excessive)")

# 🎯 Prédiction + stockage dans session_state
if st.button("🔍 LANCER L'ANALYSE", use_container_width=True):
    input_df = pd.DataFrame({
        'Age': [age],
        'Gender': [1 if gender == "Masculin" else 0],
        'Polyuria': [1 if polyuria else 0],
        'Polydipsia': [1 if polydipsia else 0],
        'sudden weight loss': [1 if weight_loss else 0],
        'partial paresis': [1 if paresis else 0],
        'Polyphagia': [1 if polyphagia else 0],
        'BMI': [bmi]
    })

    input_scaled = scaler.transform(input_df)
    st.session_state.proba = model.predict_proba(input_scaled)[0, 1]
    st.session_state.pred = 1 if st.session_state.proba >= 0.5 else 0
    st.session_state.analyse_done = True

# ✅ Affichage des résultats (hors du if)
if st.session_state.analyse_done:
    proba = st.session_state.proba
    pred = st.session_state.pred

    # --- RÉSULTATS ---
    st.markdown("---")
    col_res1, col_res2 = st.columns([1, 2])

    with col_res1:
        if pred == 1:
            st.markdown(
                '<div class="metric-card" style="background: linear-gradient(135deg, #ff6b6b, #ee5a52); color: white;">'
                '<h3>⚠️ Risque élevé</h3>'
                f'<h2>{proba:.1%}</h2>'
                '</div>', unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="metric-card" style="background: linear-gradient(135deg, #51cf66, #40c057); color: white;">'
                '<h3>✅ Risque faible</h3>'
                f'<h2>{(1-proba):.1%}</h2>'
                '</div>', unsafe_allow_html=True
            )

    with col_res2:
        fig, ax = plt.subplots(figsize=(4, 4), facecolor='none')
        wedges, texts, autotexts = ax.pie(
            [1 - proba, proba],
            labels=['Sain', 'Diabète'],
            colors=['#51cf66', '#ff6b6b'],
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops=dict(width=0.5, edgecolor='w')
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax.axis('equal')
        st.pyplot(fig)

    # --- RECOMMANDATIONS ---
    st.markdown("### 📋 Recommandations")
    if pred == 1:
        st.warning("**1.** Consultez un médecin rapidement pour un bilan glycémique complet.")
        st.warning("**2.** Contrôlez votre alimentation : réduisez les sucres simples.")
        st.warning("**3.** Pratiquez une activité physique régulière (30 min/jour).")
    else:
        st.success("**1.** Maintenez un mode de vie équilibré.")
        st.success("**2.** Faites des contrôles annuels.")
        st.success("**3.** Gardez une alimentation riche en fibres et pauvre en sucres.")


            

# --- BOUTON TELECHARGER PDF (toujours visible si analyse faite) ---
if st.session_state.analyse_done:
    st.markdown("---")
    st.markdown("### 📥 Télécharger le rapport")

    from fpdf import FPDF

    def create_pdf(age, gender, bmi, proba, pred, polyuria, polydipsia, weight_loss, paresis, polyphagia):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        pdf.cell(0, 10, text="DiabCheck Pro - Rapport de diagnostic", new_x="LMARGIN", new_y="NEXT", align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, text=f"Date : {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        pdf.cell(0, 10, text="Données patient :", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 10, text=f"Âge : {age} ans", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 10, text=f"Genre : {gender}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 10, text=f"IMC : {bmi} kg/m²", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        pdf.cell(0, 10, text="Symptômes :", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 10, text=f"Polyurie : {'Oui' if polyuria else 'Non'}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 10, text=f"Polydipsie : {'Oui' if polydipsia else 'Non'}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 10, text=f"Perte de poids : {'Oui' if weight_loss else 'Non'}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 10, text=f"Paresthésie : {'Oui' if paresis else 'Non'}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 10, text=f"Polyphagie : {'Oui' if polyphagia else 'Non'}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        pdf.cell(0, 10, text="Résultat :", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 10, text=f"Risque de diabète : {proba:.1%}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 10, text=f"Recommandation : {'Consulter rapidement' if pred else 'Continuer contrôles'}", new_x="LMARGIN", new_y="NEXT")
        return bytes(pdf.output())

    proba = st.session_state.proba
    pred = st.session_state.pred
    pdf_bytes = create_pdf(age, gender, bmi, proba, pred, polyuria, polydipsia, weight_loss, paresis, polyphagia)
    st.download_button(
        label="📄 Télécharger rapport PDF",
        data=pdf_bytes,
        file_name=f"rapport_diabete_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 📊 Performance du modèle")
    st.metric("Précision", "95.2%", delta="+2.1%")
    st.metric("Rappel", "95.3%", delta="+1.8%")
    st.metric("F1-Score", "95.2%", delta="+2.0%")

    st.markdown("### 🎯 Features les + importants")
    st.markdown("""
    1. **Polyurie** (urination fréquente)
    2. **Polydipsie** (soif excessive)
    3. **Perte de poids soudaine**
    4. **Engourdissement partiel**
    """)

    st.markdown("### 📞 Contact urgence")
    st.markdown("**☎ 190** : Numéro national santé (Tunisie)")
    st.markdown("**🕒 24h/24 - 7j/7**")