# Image Captioning with CNN & LSTM
Ce projet implémente un modèle de Deep Learning capable de générer automatiquement des légendes textuelles à partir d'images. Il utilise une architecture "Encoder-Decoder" combinant la vision par ordinateur (CNN) et le traitement du langage naturel (RNN/LSTM).
# Aperçu du Modèle
Le modèle repose sur deux piliers principaux :

Encoder (CNN) : Utilisation de InceptionV3 (pré-entraîné sur ImageNet) pour extraire les caractéristiques visuelles des images (vecteurs de taille 2048).

Decoder (LSTM) : Un réseau récurrent qui prend les caractéristiques de l'image et les séquences de mots précédentes pour prédire le mot suivant de la légende.
# Installation & Configuration
* Pré-requis:

Un compte Google (pour utiliser Google Colab).

Le dataset Flickr8k (Images + Captions).

Les fichiers de features extraites (ex: image_features.pkl).

# Entraînement
L'entraînement a été optimisé avec les paramètres suivants pour éviter le bégaiement du modèle :

Optimizer : Adam avec un learning_rate=0.0001.

Loss Function : categorical_crossentropy.

Régularisation : Couches Dropout (0.2 à 0.5) pour éviter l'overfitting.

Callbacks : ReduceLROnPlateau et ModelCheckpoint pour capturer la meilleure version du modèle.

# Résultats
Le modèle est capable de passer d'une répétition simple à des phrases structurées :

* Début : startseq a a in a a endseq

* Final : a boy in a red shirt is playing with a ball

# Utilisation
Pour tester le modèle sur vos propres images :

Montez votre Google Drive.

Chargez le modèle best_model.h5.

Utilisez la fonction generate_caption() pour transformer une image en texte.
* Exemple rapide
description = generate_caption(feature_vector, model, word_to_index, index_to_word, max_length)
print(clean_captionn(description))

# Améliorations futures
[] Implémenter le Beam Search pour une meilleure précision.

[ ] Utiliser un dataset plus large (Flickr30k ou MSCOCO).

[ ] Ajouter un mécanisme d'Attention pour que le modèle se focalise sur des zones précises de l'image.

# About
Ce projet a été réalisé dans le cadre de mon apprentissage du Deep Learning et de la Computer Vision, avec un focus particulier sur les architectures hybrides CNN-LSTM.
