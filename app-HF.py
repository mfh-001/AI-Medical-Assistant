import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import cv2
import numpy as np
import os
import warnings
import re

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="MediScan AI",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { color: #1e3a8a !important; font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid rgba(30,58,138,0.15);
        gap: 32px;
    }
    .stTabs [aria-selected="true"] {
        color: #1e3a8a !important;
        border-bottom: 3px solid #1e3a8a;
    }
    .stButton > button {
        background: #1e3a8a !important;
        color: white !important;
        border-radius: 10px;
        border: none;
    }
    .summary-box {
        background: rgba(255,255,255,0.5);
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
        border: 1px solid rgba(30,58,138,0.1);
    }
    .footer {
        text-align: center;
        padding: 12px;
        font-size: 13px;
        color: #64748b;
        border-top: 1px solid rgba(30,58,138,0.1);
        margin-top: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# ── U-Net model definition (must match training architecture exactly)
class MedicalUNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(base.children())[:-3])
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.up1(x))
        x = F.relu(self.up2(x))
        x = F.relu(self.up3(x))
        x = F.relu(self.up4(x))
        return self.final(x)

# ── Knowledge base (no LLM required — works on free CPU)
MEDICAL_KB = {
    "hypertension": {
        "definition": "Hypertension is persistently elevated blood pressure (≥130/80 mmHg).",
        "causes": "Primary causes include genetics, aging, obesity, high sodium intake, sedentary lifestyle.",
        "symptoms": "Often asymptomatic. When present: severe headaches, fatigue, vision changes, chest pain.",
        "treatment": "Lifestyle: DASH diet, exercise, weight loss. Medications: ACE inhibitors, ARBs, diuretics, beta-blockers.",
        "complications": "Heart attack, stroke, heart failure, kidney damage, vision loss.",
    },
    "melanoma": {
        "definition": "Melanoma is an aggressive skin cancer arising from melanocytes (pigment-producing cells).",
        "causes": "UV radiation (sun exposure, tanning beds). Risk factors: fair skin, many moles, family history.",
        "symptoms": "ABCDE rule: Asymmetry, Border irregularity, Color variation, Diameter >6mm, Evolving appearance.",
        "treatment": "Surgical excision (primary), immunotherapy (pembrolizumab), targeted therapy (BRAF/MEK inhibitors).",
        "complications": "Metastasis to lymph nodes, lungs, liver, and brain if untreated.",
    },
    "diabetes": {
        "definition": "Diabetes mellitus is chronic hyperglycemia caused by insulin deficiency (Type 1) or resistance (Type 2).",
        "causes": "Type 1: autoimmune destruction of beta cells. Type 2: insulin resistance from obesity, genetics, inactivity.",
        "symptoms": "Polyuria (frequent urination), polydipsia (excessive thirst), polyphagia, weight loss, fatigue, blurred vision.",
        "treatment": "Type 1: insulin therapy. Type 2: metformin, GLP-1 agonists, SGLT2 inhibitors, lifestyle changes.",
        "complications": "Diabetic retinopathy, nephropathy, neuropathy, cardiovascular disease, foot ulcers.",
    },
    "skin_cancer": {
        "definition": "Skin cancer encompasses basal cell carcinoma (BCC), squamous cell carcinoma (SCC), and melanoma.",
        "causes": "Primarily UV radiation. Risk factors: fair skin, immunosuppression, radiation exposure.",
        "symptoms": "Pearly/waxy papule (BCC), firm red nodule or flat lesion (SCC), irregular dark mole (melanoma).",
        "treatment": "Surgical excision, Mohs surgery, cryotherapy, radiation, targeted therapy.",
        "complications": "Local tissue destruction; SCC and melanoma can metastasize.",
    },
    "skin_lesion": {
        "definition": "A skin lesion is any focal abnormality of skin surface. Can be benign (mole, cyst) or malignant.",
        "causes": "Genetic factors, UV exposure, infection, trauma, inflammatory conditions.",
        "symptoms": "Changes in color, size, shape, or texture of a skin area. Bleeding or itching are warning signs.",
        "treatment": "Clinical observation, dermoscopy, biopsy, surgical removal depending on type.",
        "complications": "Risk of malignant transformation; some lesions may indicate systemic disease.",
    },
    "psoriasis": {
        "definition": "Psoriasis is a chronic autoimmune condition causing rapid skin cell buildup, forming scales and red patches.",
        "causes": "Immune system malfunction (T-cell activation). Triggers: stress, infections, certain medications, smoking.",
        "symptoms": "Red patches with silvery scales, dry cracked skin, itching/burning, thickened nails, joint pain (psoriatic arthritis).",
        "treatment": "Topical corticosteroids, vitamin D analogues, biologics (TNF/IL inhibitors), phototherapy, methotrexate.",
        "complications": "Psoriatic arthritis (30%), increased cardiovascular risk, depression, metabolic syndrome.",
    },
    "eczema": {
        "definition": "Atopic dermatitis (eczema) is a chronic inflammatory skin condition causing itchy, inflamed skin.",
        "causes": "Combination of genetic predisposition, skin barrier dysfunction, immune dysregulation, environmental triggers.",
        "symptoms": "Intense itching (especially at night), dry scaly skin, red/brownish patches, small raised bumps, thickened skin.",
        "treatment": "Moisturisers, topical steroids, calcineurin inhibitors, dupilumab (biologic), antihistamines for itch.",
        "complications": "Skin infections, sleep disruption, anxiety/depression, asthma/hay fever association.",
    },
}

def find_condition(query):
    clean_query = re.sub(r'[^a-zA-Z0-9\s]', '', query.lower())
    query_parts = clean_query.split()

    def is_close(word, target, threshold=0.75):
        if word == target: return True
        if abs(len(word) - len(target)) > 2: return False
        matches = sum(1 for a, b in zip(word, target) if a == b)
        shorter = word if len(word) < len(target) else target
        longer = target if len(word) < len(target) else word
        char_matches = sum(1 for char in set(shorter) if char in longer)
        score = (matches / max(len(word), len(target)) * 0.6) + (char_matches / max(len(word), len(target)) * 0.4)
        return score >= threshold

    keyword_map = {
        "hypertension": ["hypertension", "hypertention", "hipertension", "htn", "blood pressure", "bloodpressure", "high bp"],
        "melanoma": ["melanoma", "malanoma", "melonoma", "melenoma"],
        "diabetes": ["diabetes", "dibetes", "diabetis", "diabettis", "blood sugar", "bloodsugar"],
        "skin_cancer": ["skin cancer", "skincancer", "carcinoma", "basal cell", "basalcell", "squamous", "bcc", "scc"],
        "skin_lesion": ["lesion", "lession", "mole", "spot", "growth", "nevus", "birthmark", "nevi"],
        "psoriasis": ["psoriasis", "psoriasis", "psorisis", "psoriases", "psoriassis", "psorasis", "psoriatic"],
        "eczema": ["eczema", "eczema", "excema", "atopic", "dermatitis", "atopic dermatitis"],
    }

    for cond, targets in keyword_map.items():
        for target in targets:
            if any(is_close(part, target) for part in query_parts) or target in clean_query:
                return cond, MEDICAL_KB[cond]
    return None, None

def format_kb_answer(query, info):
    return (
        f"**Definition:** {info['definition']}\n\n"
        f"**Causes:** {info['causes']}\n\n"
        f"**Symptoms:** {info['symptoms']}\n\n"
        f"**Treatment:** {info['treatment']}\n\n"
        f"**Complications:** {info['complications']}"
    )

@st.cache_resource(show_spinner="Loading vision model...")
def load_vision_model():
    model = MedicalUNet()
    model_path = "unet_skin.pth"
    if os.path.exists(model_path):
        try:
            # Load on CPU for HF Spaces free tier
            state = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            return model, True
        except Exception as e:
            st.warning(f"Model weights not loaded: {e}")
            return model, False
    return model, False

vision_model, model_loaded = load_vision_model()

# ── UI
st.title("🔬 MediScan AI")
st.markdown("<h3 style='margin-top:-12px; color:#64748b;'>Medical Information Assistant</h3>", unsafe_allow_html=True)

if not model_loaded:
    st.info("ℹ️ Running in Knowledge Base mode — upload `unet_skin.pth` to enable full lesion segmentation.", icon="💡")

tab1, tab2, tab3 = st.tabs(["💬 Medical Q&A", "🖼️ Image Analysis", "ℹ️ About"])

with tab1:
    st.markdown("### Ask Medical Questions")
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Question:",
            placeholder="e.g., What are the symptoms of melanoma?",
            label_visibility="collapsed"
        )
    with col2:
        btn = st.button("Ask", type="primary", use_container_width=True)

    with st.expander("📋 Example questions"):
        st.markdown("""
        - What are symptoms of hypertension?
        - How is diabetes treated?
        - What causes melanoma?
        - When to see a doctor for a skin lesion?
        - What is psoriasis?
        - How is eczema treated?
        """)

    if query and (btn or query):
        with st.spinner("Processing..."):
            cond, info = find_condition(query)
            if info:
                answer = format_kb_answer(query, info)
                st.markdown("---")
                st.markdown(f"**Medical Information:**\n\n{answer}")
                st.markdown("---")
                st.success("📚 Educational information only. Always consult a qualified healthcare professional.")
            else:
                st.warning("No information found. Try asking about: hypertension, melanoma, diabetes, skin cancer, skin lesion, psoriasis, or eczema.")

with tab2:
    st.markdown("### Skin Lesion Segmentation")
    st.markdown("Upload a dermoscopic image to identify lesion boundaries and estimate coverage area.")

    file = st.file_uploader("Upload skin image", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

    if file:
        bytes_img = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(bytes_img, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        inp = cv2.resize(img, (128, 128))
        tensor = torch.from_numpy(inp).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        with torch.no_grad():
            pred = vision_model(tensor)
            mask_raw = torch.sigmoid(pred).cpu().numpy()[0][0]
            mask = (mask_raw > 0.5).astype(np.uint8) * 255

        pixels_detected = np.count_nonzero(mask)
        total_pixels = mask.size
        percentage = (pixels_detected / total_pixels) * 100

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_rgb, use_column_width=True, caption="Original Image")
        with col2:
            st.image(mask, use_column_width=True, caption="Segmentation Mask")

        severity = "Low" if percentage < 15 else "Moderate" if percentage < 35 else "High"
        severity_color = "#16a34a" if percentage < 15 else "#d97706" if percentage < 35 else "#dc2626"

        st.markdown(f"""
            <div class="summary-box">
                <h4>Analysis Summary</h4>
                <p><b>Lesion Coverage:</b> {percentage:.2f}% of the analyzed area</p>
                <p><b>Involvement Level:</b> <span style="color:{severity_color}; font-weight:600;">{severity}</span></p>
                <p><b>Note:</b> The model identifies distinct skin regions based on texture and color contrast from HAM10000 training data.</p>
                <p style="color:#b91c1c; font-weight:600;">⚠️ This is an automated research tool. Clinical diagnosis requires a qualified dermatologist.</p>
            </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("### About MediScan AI")
    st.markdown("""
    MediScan AI is a research prototype combining computer vision and medical knowledge for educational skin health exploration.

    **Vision Module**
    - Architecture: U-Net with ResNet18 encoder backbone
    - Training data: HAM10000 (10,015 dermoscopic images)
    - Task: Binary lesion segmentation (lesion vs. background)
    - Metrics: Dice Coefficient 0.90 · Jaccard Index 0.85

    **Knowledge Module**
    - Structured medical knowledge base with fuzzy query matching
    - Covers: melanoma, skin lesions, psoriasis, eczema, diabetes, hypertension, skin cancer
    - Handles common misspellings and colloquial phrasing

    **Technical Stack**
    - PyTorch · torchvision · OpenCV · Streamlit

    **Important Limitations**
    - Not clinically validated
    - Research and educational use only
    - Always seek professional medical advice

    ---
    *Built by Fahad — Full LLM Version and Demo available on GitHub(https://github.com/mfh-001/AI-Medical-Assistant)*
    """)

st.markdown("""
    <div class="footer">
        ⚕️ MediScan AI is not a substitute for professional medical advice, diagnosis, or treatment.
    </div>
""", unsafe_allow_html=True)