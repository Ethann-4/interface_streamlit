# detect.py
import os
import cv2
import numpy as np
from ultralytics import YOLO

def run_detection(image_path, model_path, output_dir):
    """
    Exécute la détection d'objets avec segmentation et dessine les contours sur une image.

    Args:
        image_path (str): Chemin vers l'image à traiter.
        model_path (str): Chemin vers le fichier de poids du modèle YOLO (best.pt),
                          doit être un modèle entraîné pour la segmentation.
        output_dir (str): Chemin vers le répertoire où enregistrer l'image annotée.

    Returns:
        str: Chemin vers l'image annotée si la détection réussit, sinon None.
    """
    # Charger le modèle YOLO (doit être un modèle de segmentation)
    try:
        model = YOLO(model_path)
        # Vérifier si le modèle est de type segmentation (optionnel mais recommandé)
        # if not hasattr(model.model, 'predict_segment'):
        #      print(f"Erreur: Le modèle à {model_path} n'est pas un modèle de segmentation.")
        #      return None
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return None

    # Charger l'image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image introuvable : {image_path}")
        return None

    print(f"Traitement de l'image pour segmentation : {os.path.basename(image_path)}")

    # Utiliser le modèle YOLO pour détecter et segmenter les objets
    try:
        # Utiliser l'image directement pour l'inférence, pas seulement le chemin
        results = model(img, verbose=False)
    except Exception as e:
        print(f"Erreur lors de l'inférence : {e}")
        return None

    # Créer une copie de l'image pour dessiner les contours
    img_with_contours = img.copy()

    # Définir les couleurs spécifiques pour chaque classe (copié de votre notebook)
    # Assurez-vous que l'ordre des classes correspond à celui de votre data.yaml
    # dans votre data.yaml, les classes sont : ['colonne_cervicale', 'epiglotte', 'langue', 'os_hyoide', 'pharynx']
    # Le mapping par défaut de YOLO commence à 0.
    # 0: 'colonne_cervicale'
    # 1: 'epiglotte'
    # 2: 'langue'
    # 3: 'os_hyoide'
    # 4: 'pharynx'
    class_colors = {
        0: (255, 0, 0),   # Bleu pour colonne_cervicale
        1: (255, 255, 0), # Jaune pour epiglotte
        2: (0, 255, 0),   # Vert pour langue
        3: (0, 255, 0),   # Vert pour os_hyoide (couleur par défaut si non spécifié ci-dessous)
        4: (0, 0, 255)    # Rouge pour pharynx
    }
    # Si tu as des couleurs spécifiques pour os_hyoide, tu peux les ajouter ou modifier ci-dessus.
    # Par exemple: class_colors[3] = (Votre_couleur_pour_os_hyoide)


    # Afficher les résultats de la segmentation et tracer les contours
    # Vérifier s'il y a des masques dans les résultats (pour la segmentation)
    if results and results[0].masks is not None:
        for result in results:
            masks = result.masks.data.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for mask, class_id in zip(masks, classes):
                class_id = int(class_id)

                # S'assurer que la classe ID est dans notre dictionnaire de couleurs
                if class_id in class_colors:
                    color = class_colors[class_id]
                else:
                    # Couleur aléatoire si la classe n'est pas dans notre liste définie
                    color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                    print(f"Classe ID {class_id} non trouvée dans les couleurs définies, utilisant une couleur aléatoire.")


                # Redimensionner le masque pour correspondre à l'image originale
                # Utiliser cv2.INTER_NEAREST pour éviter l'interpolation qui peut lisser les contours
                mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)


                # Convertir le masque redimensionné en format binaire
                # Utiliser un seuil, par exemple 0.5, pour les masques flottants
                binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255

                # Trouver les contours sur le masque binaire
                # cv2.RETR_EXTERNAL pour trouver uniquement les contours extérieurs (souvent suffisant)
                # cv2.CHAIN_APPROX_SIMPLE pour compresser les points redondants
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


                # Dessiner les contours sur l'image
                # -1 pour dessiner tous les contours trouvés pour ce masque
                cv2.drawContours(img_with_contours, contours, -1, color, 2) # 2 est l'épaisseur de la ligne

    else:
        print(f"Aucun masque de segmentation trouvé dans les résultats pour l'image : {os.path.basename(image_path)}. Le modèle pourrait ne pas être un modèle de segmentation ou aucune détection n'a été faite.")
        # Tu peux choisir de retourner l'image originale ou None si aucune segmentation n'est trouvée
        # Pour l'instant, on retourne l'image originale (copie) même sans contours dessinés.


    # Enregistrer l'image avec les contours dessinés
    output_image_name = f"annotated_contours_{os.path.basename(image_path)}"
    # Utiliser l'extension .png car les masques peuvent avoir des bords irréguliers,
    # et le format PNG est sans perte.
    output_path = os.path.join(output_dir, output_image_name)

    try:
        # Assurez-vous que le répertoire de sortie existe déjà (il est créé dans app.py)
        cv2.imwrite(output_path, img_with_contours)
        print(f"Image avec contours enregistrée à : {output_path}")
        return output_path
    except Exception as e:
        print(f"Erreur lors de l'enregistrement de l'image annotée avec contours : {e}")
        return None

# Note: La partie __main__ n'est pas utilisée lorsque ce script est importé.
