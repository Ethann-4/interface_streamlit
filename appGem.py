import streamlit as st
import os
import cv2
from PIL import Image
import numpy as np
import zipfile
import shutil

# Assurez-vous que detect.py est dans le même répertoire ou accessible
from detect import run_detection

st.title("Détection de Structures Anatomiques avec YOLOv12")

st.write("""

Cette application vous permet de télécharger un fichier ZIP contenant des images (même dans des sous-dossiers),
de sélectionner un modèle YOLOv12 entraîné, et d'exécuter la détection d'objets sur toutes les images.
""")

st.sidebar.header("Configuration de la Détection")

# Section pour le téléchargement d'un fichier ZIP
st.sidebar.subheader("Télécharger un dossier d'images (fichier ZIP)")
uploaded_zip_file = st.sidebar.file_uploader("Choisissez un fichier ZIP...", type=["zip"])

# Chemin vers votre modèle YOLO
MODEL_PATH = "yolov12/models/best.pt" # Ajustez ce chemin si nécessaire

# Répertoires pour enregistrer temporairement les fichiers
UPLOAD_DIR = "uploaded_data"
EXTRACTED_DIR = os.path.join(UPLOAD_DIR, "extracted_content") # Renommé pour clarifier que c'est le contenu du ZIP
ANNOTATED_IMAGES_DIR = os.path.join(UPLOAD_DIR, "annotated_images")

# Créer les répertoires nécessaires
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXTRACTED_DIR, exist_ok=True)
os.makedirs(ANNOTATED_IMAGES_DIR, exist_ok=True)

# Variable pour stocker la liste des chemins complets des images trouvées
image_file_paths = []

if uploaded_zip_file is not None:
    # Enregistrer temporairement le fichier ZIP téléchargé
    zip_file_path = os.path.join(UPLOAD_DIR, uploaded_zip_file.name)
    with open(zip_file_path, "wb") as f:
        f.write(uploaded_zip_file.getbuffer())

    st.sidebar.success("Fichier ZIP téléchargé avec succès.")
    st.write("Extraction du fichier ZIP...")

    # Nettoyer le répertoire d'extraction avant d'extraire le nouveau ZIP
    if os.path.exists(EXTRACTED_DIR):
        shutil.rmtree(EXTRACTED_DIR)
    os.makedirs(EXTRACTED_DIR, exist_ok=True)

    # Décompresser le fichier ZIP
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(EXTRACTED_DIR)
        st.write("Extraction terminée.")

        # Parcourir le répertoire extrait pour trouver toutes les images
        image_file_paths = [] # Réinitialiser la liste des chemins d'images
        valid_image_extensions = ('.jpg', '.jpeg', '.png', '.webp')

        st.write(f"Recherche d'images dans le contenu extrait ({EXTRACTED_DIR})...")
        for root, dirs, files in os.walk(EXTRACTED_DIR):
            for file in files:
                if file.lower().endswith(valid_image_extensions):
                    # Stocker le chemin complet de l'image
                    image_path = os.path.join(root, file)
                    image_file_paths.append(image_path)

        st.write(f"Nombre total d'images valides trouvées : {len(image_file_paths)}")

        if len(image_file_paths) > 0:
             st.write("Quelques images trouvées (les 5 premières ou moins):")
             # Afficher les 5 premières images trouvées (ou moins s'il y en a moins de 5)
             for i, img_path in enumerate(image_file_paths[:min(len(image_file_paths), 5)]):
                 try:
                     image = Image.open(img_path)
                     st.image(image, caption=f"Image : {os.path.basename(img_path)}", width=150)
                 except Exception as e:
                     st.warning(f"Impossible d'afficher l'image {os.path.basename(img_path)} (chemin: {img_path}). Erreur : {e}")
        else:
            st.warning("Aucun fichier image valide trouvé dans le fichier ZIP extrait ou dans ses sous-dossiers.")

    except Exception as e:
        st.error(f"Erreur lors de l'extraction du fichier ZIP : {e}")
        image_file_paths = [] # Réinitialiser la liste

    # Nettoyer le fichier ZIP après extraction (optionnel)
    # os.remove(zip_file_path)


# Bouton pour lancer la détection
st.sidebar.write("---") # Ajoute une ligne de séparation
if st.sidebar.button("Lancer la détection sur les images trouvées"):
    if image_file_paths: # Vérifier si la liste des chemins d'images n'est pas vide
        st.write("Détection en cours sur toutes les images...")

        # Nettoyer le répertoire des images annotées avant de commencer
        if os.path.exists(ANNOTATED_IMAGES_DIR):
             shutil.rmtree(ANNOTATED_IMAGES_DIR)
        os.makedirs(ANNOTATED_IMAGES_DIR, exist_ok=True)

        progress_bar = st.progress(0)
        num_processed = 0
        total_images = len(image_file_paths)
        successful_annotations = [] # Pour garder la trace des chemins des images annotées avec succès

        for i, image_path in enumerate(image_file_paths):
            st.write(f"Traitement de l'image {i+1}/{total_images} : {os.path.basename(image_path)}") # Indicateur de progression textuel

            # Exécuter la détection sur chaque image
            try:
                annotated_image_path = run_detection(image_path, MODEL_PATH, ANNOTATED_IMAGES_DIR)

                if annotated_image_path and os.path.exists(annotated_image_path):
                    num_processed += 1
                    progress_bar.progress(num_processed / total_images)
                    successful_annotations.append(annotated_image_path) # Ajouter le chemin à la liste

            except Exception as e:
                 st.error(f"Erreur lors du traitement de l'image {os.path.basename(image_path)} : {e}")

        st.success(f"Détection terminée sur {num_processed} images sur un total de {total_images} images trouvées.")

        # --- NOUVEAU : Afficher toutes les images annotées à la fin ---
        if successful_annotations:
            st.write("---") # Séparation visuelle
            st.subheader("Images annotées :")

            # Utiliser st.columns pour afficher les images en grille si tu veux
            # Par exemple, afficher 3 images par ligne
            cols = st.columns(3)
            for i, annotated_img_path in enumerate(successful_annotations):
                try:
                    annotated_image = Image.open(annotated_img_path)
                    # Afficher dans la colonne appropriée
                    with cols[i % 3]: # Utiliser l'opérateur modulo pour cycler à travers les colonnes
                        st.image(annotated_image, caption=f"Annoté : {os.path.basename(annotated_img_path)}", width=200) # Ajuste la largeur si nécessaire
                except Exception as e:
                     st.warning(f"Impossible d'afficher l'image annotée {os.path.basename(annotated_img_path)}. Erreur : {e}")

        else:
            st.write("Aucune image n'a été annotée avec succès.")
        # --- FIN NOUVEAU ---


        # --- AJOUT POUR LE TÉLÉCHARGEMENT DU DOSSIER ANNOTÉ ---
        # Créer un fichier ZIP contenant les images annotées
        annotated_zip_filename = "annotated_images.zip"
        annotated_zip_path = os.path.join(UPLOAD_DIR, annotated_zip_filename)

        if successful_annotations: # Vérifier s'il y a des images annotées avec succès pour le zip
             try:
                 with zipfile.ZipFile(annotated_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                     for full_path in successful_annotations:
                         zipf.write(full_path, os.path.relpath(full_path, ANNOTATED_IMAGES_DIR))

                 # Offrir le téléchargement du fichier ZIP annoté
                 with open(annotated_zip_path, "rb") as file:
                      btn = st.download_button(
                          label="Télécharger toutes les images annotées (ZIP)",
                          data=file,
                          file_name=annotated_zip_filename,
                          mime="application/zip"
                      )

             except Exception as e:
                  st.error(f"Erreur lors de la création du fichier ZIP annoté : {e}")

        else:
            st.warning("Aucune image annotée disponible pour la création du fichier ZIP.")

        # Nettoyer les répertoires temporaires (optionnel mais recommandé)
        # shutil.rmtree(EXTRACTED_DIR)
        # shutil.rmtree(ANNOTATED_IMAGES_DIR)
        # image_file_paths = [] # Réinitialiser la liste

    else:
        st.sidebar.warning("Veuillez d'abord télécharger et extraire un fichier ZIP contenant des images.")

# Vous pouvez ajouter d'autres sections ou informations ici
st.write("---")
st.write("Développé avec Streamlit et YOLOv12.")