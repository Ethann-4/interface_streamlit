# app.py
import streamlit as st
import os
import cv2
from PIL import Image
import numpy as np

# Assurez-vous que detect.py est dans le même répertoire ou accessible
from detect import run_detection

# Chemin vers votre modèle YOLO
MODEL_PATH = "yolov12/models/best.pt" # Ajustez ce chemin si nécessaire

# Chemin pour enregistrer temporairement les images téléchargées et annotées
UPLOAD_DIR = "uploaded_images"
ANNOTATED_DIR = "annotated_images"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)

st.title("Détection d'Structures Anatomiques avec YOLOv12")

st.sidebar.header("Paramètres")
# Vous pouvez ajouter des paramètres ici si nécessaire, par exemple un seuil de confiance

uploaded_file = st.file_uploader("Téléchargez une image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Afficher l'image téléchargée
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée.", use_column_width=True)

    # Enregistrer temporairement l'image téléchargée
    image_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("")
    st.write("Détection en cours...")

    # Exécuter la détection
    annotated_image_path = run_detection(image_path, MODEL_PATH, ANNOTATED_DIR)

    if annotated_image_path:
        # Afficher l'image avec les annotations
        annotated_image = Image.open(annotated_image_path)
        st.image(annotated_image, caption="Image avec détections.", use_column_width=True)

        # Optionnel : Afficher les résultats de la détection (classes, confidences, etc.)
        # Vous devrez modifier run_detection pour retourner ces informations si vous les voulez ici

    else:
        st.error("La détection a échoué.")

    # Nettoyer les fichiers temporaires (optionnel mais recommandé)
    # os.remove(image_path)
    # if annotated_image_path and os.path.exists(annotated_image_path):
    #     os.remove(annotated_image_path)
