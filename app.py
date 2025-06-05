import streamlit as st
import os
from PIL import Image
import shutil
import subprocess
import uuid

# Répertoires
UPLOAD_DIR = "uploads"
RESULTS_DIR = "runs/detect"

# Crée les dossiers nécessaires
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Interface
st.set_page_config(page_title="Détection Médicale - YOLOv12", layout="wide")
st.title("Analyse Médicale par Détection Automatique - YOLOv12")

st.sidebar.header("📁 Paramètres de détection")
model_path = st.sidebar.selectbox(
    "Choisir le modèle YOLOv12",
    ["best.pt"],  # Tu peux ajouter d'autres modèles si besoin
    index=0
)

conf_thres = st.sidebar.slider("🔍 Seuil de confiance", 0.0, 1.0, 0.25, 0.01)
iou_thres = st.sidebar.slider("📐 Seuil IoU", 0.0, 1.0, 0.45, 0.01)

st.sidebar.markdown("---")
st.sidebar.info("💡 Charge une image médicale pour détecter les structures.")

# Upload image
uploaded_file = st.file_uploader("📤 Importer une image (jpg, png)", type=["jpg", "jpeg", "png"])
run_detection = st.button("🚀 Lancer la détection")

if uploaded_file is not None:
    file_id = str(uuid.uuid4())
    image_path = os.path.join(UPLOAD_DIR, f"{file_id}_{uploaded_file.name}")
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(Image.open(image_path), caption="Image importée", use_container_width=True)

    if run_detection:
        st.subheader(" Résultats de la détection")

        # Commande pour exécuter detect.py
        command = [
            "python",
            "yolov12/detect.py",
            "--weights", model_path,
            "--source", image_path,
            "--conf-thres", str(conf_thres),
            "--iou-thres", str(iou_thres),
            "--save-txt",
            "--save-conf"
        ]

        with st.spinner("🧠 Détection en cours..."):
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Affichage des logs
        with st.expander("📋 Voir les logs de détection"):
            st.code(result.stdout + "\n" + result.stderr)

        # Cherche la dernière image générée
        result_folders = sorted(os.listdir(RESULTS_DIR), key=lambda x: os.path.getctime(os.path.join(RESULTS_DIR, x)), reverse=True)
        if result_folders:
            last_run = result_folders[0]
            result_path = os.path.join(RESULTS_DIR, last_run)
            output_image_path = os.path.join(result_path, os.path.basename(image_path))

            if os.path.exists(output_image_path):
                st.image(output_image_path, caption="🧠 Résultat de la détection", use_container_width=True)
            else:
                st.warning("❌ Image détectée non trouvée.")
        else:
            st.error("❌ Aucun résultat généré.")
