import streamlit as st
import os
import cv2
from PIL import Image
import numpy as np

# Assurez-vous que detect.py est dans le même répertoire ou accessible
from detect import run_detection # Nous allons utiliser la fonction run_detection que nous avons définie précédemment

st.title("Détection d'Structures Anatomiques avec YOLOv12")

st.write("""
## Interface GUI interactive pour l'analyse d'images avec détection YOLOv12

Cette application vous permet de télécharger une image, de sélectionner un modèle YOLOv12 entraîné,
et d'exécuter la détection d'objets pour identifier des structures anatomiques.
""")

st.sidebar.header("Configuration de la Détection")

# Section pour le téléchargement d'image
st.sidebar.subheader("Télécharger une image")
uploaded_file = st.sidebar.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png", "webp"])

# Section pour la sélection du modèle (si vous avez plusieurs modèles)
# Pour l'instant, nous utiliserons le chemin vers le modèle best.pt comme défini plus bas
# st.sidebar.subheader("Sélectionner un modèle")
# model_choice = st.sidebar.selectbox("Modèle à utiliser:", ["best.pt"]) # Vous pouvez ajouter d'autres options si vous avez plusieurs modèles

# Chemin vers votre modèle YOLO
# Assurez-vous que ce chemin est correct par rapport à l'emplacement de votre app.py
MODEL_PATH = "yolov12/models/best.pt" # Ajustez ce chemin si nécessaire

# Répertoires pour enregistrer temporairement les fichiers
UPLOAD_DIR = "uploaded_images"
ANNOTATED_DIR = "annotated_images"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)

# Variable pour stocker le chemin de l'image téléchargée pour un usage ultérieur
current_image_path = None

if uploaded_file is not None:
    # Afficher l'image téléchargée dans la zone principale
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée.", use_container_width=True)

    # Enregistrer temporairement l'image téléchargée
    image_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    current_image_path = image_path # Stocker le chemin de l'image téléchargée

    st.success("Image téléchargée avec succès !")

# Bouton pour lancer la détection
st.sidebar.write("---") # Ajoute une ligne de séparation
if st.sidebar.button("Lancer la détection"):
    if current_image_path:
        st.write("Détection en cours...")

        # Exécuter la détection en utilisant la fonction de detect.py
        # Passez le chemin de l'image téléchargée et le chemin du modèle
        annotated_image_path = run_detection(current_image_path, MODEL_PATH, ANNOTATED_DIR)

        if annotated_image_path and os.path.exists(annotated_image_path):
            # Afficher l'image avec les annotations
            annotated_image = Image.open(annotated_image_path)
            st.image(annotated_image, caption="Image avec détections.", use_container_width=True)
            st.success("Détection terminée !")

            # Nettoyer le fichier annoté temporaire (optionnel)
            # os.remove(annotated_image_path)
        else:
            st.error("La détection a échoué. Veuillez vérifier le chemin du modèle et l'image.")

        # Nettoyer le fichier image téléchargé temporaire
        # os.remove(current_image_path)
        current_image_path = None # Réinitialiser le chemin de l'image courante
    else:
        st.sidebar.warning("Veuillez d'abord télécharger une image.")

# Vous pouvez ajouter d'autres sections ou informations ici
st.write("---")
st.write("Développé avec Streamlit et YOLOv12.")
