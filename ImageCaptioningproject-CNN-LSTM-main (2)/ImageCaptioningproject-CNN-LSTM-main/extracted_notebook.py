!pip install matplotlib pillow

BASE_PATH = r"C:/Users/GIGABYTE/Downloads/ImageCaptioningproject-CNN-LSTM-main (2)"
IMAGES_PATH = os.path.join(BASE_PATH, "Images")

file_path = "C:/Users/GIGABYTE/Downloads/archive/captions.txt"

captions = {}
with open('C:/Users/GIGABYTE/Downloads/archive/captions.txt', 'r') as f:
  for line in f:
    parts=line.strip().split(',')
    img=parts[0]
    cap=",".join(parts[1:])

    if img not in captions:
      captions[img]=[]
    captions[img].append(cap)

img_path = "C:/Users/GIGABYTE/Downloads/archive/Images/96420612_feb18fc6c6.jpg"

pip install pillow

pip install numpy pandas matplotlib pillow tqdm scikit-learn tensorflow keras opencv-python nltk

import os
from PIL import Image
import matplotlib.pyplot as plt

img_name = "96420612_feb18fc6c6.jpg"

img_path = r"C:/Users/GIGABYTE/Downloads/archive/Images/96420612_feb18fc6c6.jpg"


img = Image.open(img_path)

plt.imshow(img)
plt.axis('off')

print(captions.get(img_name, "NO caption"))

import string
def clean_caption(caption):
  caption=caption.lower()
  caption=caption.translate(str.maketrans('','',string.punctuation))
  return caption

for key in captions:
  captions[key]=[clean_caption(c) for c in captions[key]]
print(captions.get("84713990_d3f3cef78b.jpg"))
for key in captions:
    new_list = []
    for c in captions[key]:
        if not c.startswith("startseq"):
            c = "startseq " + c + " endseq"
        new_list.append(c)
    captions[key] = new_list
print(captions.get("84713990_d3f3cef78b.jpg"))


all_captions = []
for cap in captions.values():
  all_captions.extend(cap)
all_captions=all_captions[1:]

vocab=set() #container of unique words assign indices to words model understand numbers
for cap in all_captions:
  for word in cap.split():
    vocab.add(word)
print("vocab size:", len(vocab)) #lstm need a fixed vocabulary size



word_to_index ={word: i+1 for i, word in enumerate(vocab)} #i+a cause 0 is preserved for le padding
index_to_word={i: word for word,i in word_to_index.items()} #decode predic

def caption_to_sequence(caption):
  return [word_to_index[word] for word in caption.split()]
print(all_captions[2])
print(caption_to_sequence(all_captions[2]))

pip install tensorflow

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import os

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

model = ResNet50(
    weights='imagenet',
    include_top=False,   # 🔥 THIS IS THE FIX
    pooling='avg'        # 🔥 THIS GIVES (2048,)
)

# Print the last 5 layers to see their real names
for layer in model.layers[-5:]:
    print(layer.name)

max_length=max_length= max(len(c.split())for c in all_captions)
print(max_length)
vocab_size= len(vocab)+1
steps = len(captions)
batch_size=64

from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add
from tensorflow.keras.models import Model

# --- 1. IMAGE BRANCH (Input 1) ---
# This expects the 2048 numbers we saved earlier
inputs1 = Input(shape=(2048,), name="image_input")
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

# --- 2. TEXT BRANCH (Input 2) ---
# This expects the sequence of words (max_length = 38)
inputs2 = Input(shape=(max_length,), name="text_input")
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

# --- 3. MERGE ---
decoder1 = Add()([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

# --- 4. THE FIX ---
# We define the model with BOTH inputs in a list
caption_model = Model(inputs=[inputs1, inputs2], outputs=outputs)

caption_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(caption_model.summary())

img_path = r"C:/Users/GIGABYTE/Downloads/archive/Images/96420612_feb18fc6c6.jpg"
img= image.load_img(img_path, target_size=(224, 224))
img= image.img_to_array(img)
img= np.expand_dims(img, axis=0)
img=preprocess_input(img)

features = model.predict(img)
print(features.shape)

image_features={}
image_features[img_name]=features[0]

import os
from tqdm import tqdm  # use this instead of tqdm.notebook

images_dir = "C:/Users/GIGABYTE/Downloads/archive/Images" # local folder

image_features = {}
image_names = list(captions.keys())

print(f"Starting extraction for {len(image_names)} images...")

for img_name in tqdm(image_names):
    try:
        img_path = os.path.join(images_dir, img_name)

        # Skip if image not found (IMPORTANT)
        if not os.path.exists(img_path):
            continue

        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        features = model.predict(img, verbose=0)

        image_features[img_name] = features[0]

    except Exception as e:
        continue

print(f"\n✅ Done! Extracted features for {len(image_features)} images.")

import pickle

save_path = "image_features.pkl"

with open(save_path, "wb") as f:
    pickle.dump(image_features, f)

print(f"✅ Features saved to {save_path}")

from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences #for padding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add #softmax to convert nb to float for probability
from tensorflow.keras.callbacks import ModelCheckpoint

def data_generator(captions, image_features, word_to_index, max_length, vocab_size, batch_size):
    while True:
        x1, x2, y = [], [], []
        n = 0

        for img_name, caps in captions.items():

            if img_name not in image_features:
                continue

            for cap in caps:
                seq = [word_to_index[word] for word in cap.split() if word in word_to_index]

                for i in range(1, len(seq)):
                    in_seq = seq[:i]
                    out_seq = seq[i]

                    # pad sequence ONCE (not twice like before)
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]

                    # one-hot encode output
                    out_seq = to_categorical(out_seq, num_classes=vocab_size)

                    x1.append(image_features[img_name])
                    x2.append(in_seq)
                    y.append(out_seq)

                    n += 1

                    if n == batch_size:
                        yield ((np.array(x1), np.array(x2)), np.array(y))
                        x1, x2, y = [], [], []
                        n = 0

# Create the generator instance
limited_caption=dict(list(captions.items())[:5000])
test_gen = data_generator(limited_caption, image_features, word_to_index, max_length, vocab_size, batch_size)

# Get one single batch
inputs, outputs = next(test_gen)

print(f"Inputs shape: {inputs[0].shape}, {inputs[1].shape}")
print(f"Outputs shape: {outputs.shape}") # Should be (32, vocab_size)

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
generator= data_generator(captions, image_features, word_to_index, max_length, vocab_size, batch_size)
checkpoint = ModelCheckpoint('best_model.h5', monitor='loss', save_best_only=True, verbose=1)
# Réduit la vitesse learning rate
lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=1)
# Arrête si ça ne progresse plus du tout (overfitting)
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
history = caption_model.fit(
    generator,
    steps_per_epoch=78,
    epochs=20,
    callbacks=[checkpoint, lr_scheduler, early_stop],
    verbose=1
)

from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add
from tensorflow.keras.models import Model
import tensorflow as tf
# --- Step 1: Recreate the Model Architecture ---
# (This matches the exact architecture you used during training)
inputs1 = Input(shape=(2048,), name="image_input")
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_length,), name="text_input")
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)
decoder1 = Add()([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
caption_model = Model(inputs=[inputs1, inputs2], outputs=outputs)
caption_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# --- Step 2: Load ONLY the learned weights ---
caption_model.load_weights('best_model.h5')
# Assign it to 'model' so your downstream generation cells continue to work
model = caption_model
print("Model architecture rebuilt and weights safely loaded!")

def generate_caption(image_features, model, word_to_index, index_to_word, max_length):
  in_text="startseq"
  for i in range(max_length):
    sequence=[word_to_index[w] for w in in_text.split() if w in word_to_index]
    sequence=pad_sequences([sequence],maxlen=max_length)


    yhat = model.predict([image_features.reshape(1,2048), sequence] ,verbose=0)
    yhat=np.argmax(yhat)
    word=index_to_word.get(yhat)
    if word is None:
      break
    in_text+=" "+word
    if word=="endseq":
      break
  return in_text

img_name=list(image_features.keys())[0]
feature=image_features[img_name]
caption=generate_caption(feature, caption_model, word_to_index, index_to_word, max_length)
print("generated:", caption)

def clean_captionn(caption):
  words= caption.split()
  words= [w for w in words if w not in ["startseq", "endseq"]]
  return " ".join(words)
print("final:",clean_captionn(caption))

import random
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
# 1. Choisir 5 images au hasard dans ton dictionnaire de test
# (On utilise min() par sécurité au cas où il y aurait moins de 5 features)
num_images = min(5, len(image_features))
test_images = random.sample(list(image_features.keys()), num_images)
plt.figure(figsize=(12, 20))
# 2. Le bon chemin LOCAL de vos images sur Windows
# J'ai mis le chemin de votre dossier Images que j'ai trouvé dans vos téléchargements!
base_image_path = r"C:\Users\GIGABYTE\Downloads\ImageCaptioningproject-CNN-LSTM-main (2)\Images"
for i, img_id in enumerate(test_images):
    # Génération
    feature_vec = image_features[img_id]
    
    # On passe le feature_vec et le model (que vous venez de reconstruire)
    raw_caption = generate_caption(feature_vec, model, word_to_index, index_to_word, max_length)
    final_caption = clean_captionn(raw_caption) # Fonction de nettoyage
    # Affichage
    img_path = os.path.join(base_image_path, img_id)
    ax = plt.subplot(num_images, 1, i + 1)
    
    if os.path.exists(img_path):
        ax.imshow(Image.open(img_path))
        ax.set_title(f"ID: {img_id}\nPREDICTION: {final_caption}", fontsize=12, pad=10)
    else:
        # Sécurité: Si l'image n'est pas dans le dossier, afficher quand même la prédiction
        ax.text(0.5, 0.5, f"Image introuvable: {img_id}\n\nPREDICTION: {final_caption}", 
                ha='center', va='center', fontsize=12)
        
    ax.axis('off')
plt.tight_layout()
plt.show()

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

smooth = SmoothingFunction().method1

def evaluate_bleu(real_captions, predicted_caption):
    scores = {}

    scores["BLEU-1"] = sentence_bleu(real_captions, predicted_caption, weights=(1, 0, 0, 0), smoothing_function=smooth)
    scores["BLEU-2"] = sentence_bleu(real_captions, predicted_caption, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    scores["BLEU-3"] = sentence_bleu(real_captions, predicted_caption, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
    scores["BLEU-4"] = sentence_bleu(real_captions, predicted_caption, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    return scores

import matplotlib.pyplot as plt
real_caps = captions[img_id]
predicted = final_caption.split()
scores = evaluate_bleu(real_caps, predicted)

names = list(scores.keys())
values = list(scores.values())

plt.figure()
plt.bar(names, values)
plt.title("BLEU Score Evaluation")
plt.xlabel("BLEU Type")
plt.ylabel("Score")
plt.show()

from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def beam_search(model, image_features, word_to_index, index_to_word, max_length, beam_index=3):

    start = [word_to_index["startseq"]]
    sequences = [[start, 0.0]]  # [sequence, score]

    for _ in range(max_length):
        all_candidates = []

        for seq, score in sequences:

            # Stop expanding if already ended
            if seq[-1] == word_to_index["endseq"]:
                all_candidates.append([seq, score])
                continue

            padded = pad_sequences([seq], maxlen=max_length)

            preds = model.predict(
                [image_features.reshape(1, 2048), padded],
                verbose=0
            )[0]

            # Take top K probabilities
            top_k = np.argsort(preds)[-beam_index:]

            for w in top_k:
                new_seq = seq + [w]

                # 🔥 Repetition penalty
                repetition_penalty = 0.5 if w in seq else 1.0

                new_score = score + np.log(preds[w] * repetition_penalty + 1e-10)

                all_candidates.append([new_seq, new_score])

        # 🔥 Sort by score normalized by length
        ordered = sorted(
            all_candidates,
            key=lambda x: x[1] / len(x[0]),
            reverse=True
        )

        sequences = ordered[:beam_index]

        # 🔥 STOP if all sequences ended
        if all(seq[-1] == word_to_index["endseq"] for seq, _ in sequences):
            break

    best_seq = sequences[0][0]

    # Convert to words
    final_caption = []
    for idx in best_seq:
        word = index_to_word.get(idx)
        if word == "startseq":
            continue
        if word == "endseq":
            break
        if word:
            final_caption.append(word)

    return " ".join(final_caption)

caption = beam_search(caption_model, feature_vec, word_to_index, index_to_word, max_length, beam_index=3)
print("Generated:", caption)


# 1. Choisir 3 images au hasard
num_images = min(3, len(image_features))
test_images = random.sample(list(image_features.keys()), num_images)
# Le chemin local vers vos images
base_image_path = r"C:\Users\GIGABYTE\Downloads\ImageCaptioningproject-CNN-LSTM-main (2)\Images"
# Créer la figure graphique
plt.figure(figsize=(12, 7 * num_images))
for i, img_id in enumerate(test_images):
    feature_vec = image_features[img_id]
    
    # --- LES CORRECTIONS (THE FIXES) ---
    
    # GREEDY SEARCH (Correction: "feature_vec" en premier, "model" en second)
    raw_greedy = generate_caption(feature_vec, model, word_to_index, index_to_word, max_length)
    greedy_caption = clean_captionn(raw_greedy)
    # BEAM SEARCH ("model" en premier) - On va utiliser un beam_index de 5 pour une super qualité
    beam_caption = beam_search(model, feature_vec, word_to_index, index_to_word, max_length, beam_index=5)
    
    # -----------------------------------
    
    real_caps = captions[img_id]
    
    # Graphiques (Graphics Configuration)
    img_path = os.path.join(base_image_path, img_id)
    ax = plt.subplot(num_images, 1, i + 1)
    
    # Préparer le block de texte formaté sous le titre
    format_text = (
        f"REAL CAPTIONS: {real_caps[0]}\n\n"
        f"GREEDY PRED: {greedy_caption}\n"
        f"BEAM SEARCH: {beam_caption}"
    )
    if os.path.exists(img_path):
        ax.imshow(Image.open(img_path))
        ax.set_title(format_text, fontsize=12, pad=15, loc='left', color='darkblue')
    else:
        ax.text(0.5, 0.5, f"[Image Introuvable: {img_id}]\n\n{format_text}", 
                ha='center', va='center', fontsize=12)
        
    ax.axis('off')
plt.tight_layout()
plt.show()

all_captions = {}

for img_name, caps in captions.items():
    all_captions[img_name] = caps[0]   # take first caption only

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

corpus = list(all_captions.values())   # all captions text

tfidf_matrix = vectorizer.fit_transform(corpus)

def get_caption(image_path):
    photo = extract_features(image_path)
    
    caption = beam_search(
        model,
        photo,
        word_to_index,
        index_to_word,
        max_length,
        beam_index=3
    )
    
    return caption

from sklearn.metrics.pairwise import cosine_similarity

def find_similar_images(query_caption, top_k=5):
    query_vec = vectorizer.transform([query_caption])
    similarities = cosine_similarity(query_vec, tfidf_matrix)
    
    top_indices = similarities[0].argsort()[-top_k:][::-1]
    image_names = list(all_captions.keys())
    
    results = [image_names[i] for i in top_indices]
    return results

def text_search():
    query = input("Enter your search: ")
    
    results = find_similar_images(query)
    
    print("\nResults:")
    for r in results:
        print(r)
    
    show_results(results)

text_search()

pip install sentence-transformers

from sentence_transformers import SentenceTransformer

bert_model = SentenceTransformer('all-MiniLM-L6-v2')

corpus = list(all_captions.values())

caption_embeddings = bert_model.encode(corpus, show_progress_bar=True)

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_similar_images_bert(query, top_k=5):
    
    # Encode query
    query_embedding = bert_model.encode([query])
    
    # Compute similarity
    similarities = cosine_similarity(query_embedding, caption_embeddings)
    
    # Get top results
    top_indices = np.argsort(similarities[0])[-top_k:][::-1]
    
    image_names = list(all_captions.keys())
    results = [image_names[i] for i in top_indices]
    
    return results

def text_search_bert():
    query = input("Enter your search: ")
    
    results = find_similar_images_bert(query)
    
    print("\nResults:")
    for r in results:
        print(r)
    
    show_results(results)


def show_results(results):
    # Setup the graphic figure
    plt.figure(figsize=(15, 6))
    
    # 1. Loop through your list of image names directly
    for i, img_name in enumerate(results):
        
        # 2. Build the file path (Notice there is no ["image"] here!)
        img_path = os.path.join(
            "C:/Users/GIGABYTE/Downloads/ImageCaptioningproject-CNN-LSTM-main (2)/Images",
            img_name
        )
        
        # 3. Create a beautiful subplot for the gallery
        ax = plt.subplot(1, len(results), i + 1)
        
        if os.path.exists(img_path):
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(f"Rank #{i+1}\n{img_name}", fontsize=10, pad=10)
        else:
            ax.text(0.5, 0.5, f"Image not found:\n{img_name}", ha='center', va='center', fontsize=9)
            
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

text_search_bert()


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compare_models(query, top_k=5):
    
    image_names = list(all_captions.keys())
    
    # ===== TF-IDF =====
    query_vec_tfidf = vectorizer.transform([query])
    sim_tfidf = cosine_similarity(query_vec_tfidf, tfidf_matrix)[0]
    
    top_tfidf_idx = np.argsort(sim_tfidf)[-top_k:][::-1]
    
    
    # ===== BERT =====
    query_vec_bert = bert_model.encode([query])
    sim_bert = cosine_similarity(query_vec_bert, caption_embeddings)[0]
    
    top_bert_idx = np.argsort(sim_bert)[-top_k:][::-1]
    
    
    print("\n🔍 QUERY:", query)
    
    print("\n===== TF-IDF RESULTS =====")
    for i, idx in enumerate(top_tfidf_idx):
        print(f"{i+1}. {image_names[idx]}  | score: {sim_tfidf[idx]:.4f}")
    
    print("\n===== BERT RESULTS =====")
    for i, idx in enumerate(top_bert_idx):
        print(f"{i+1}. {image_names[idx]}  | score: {sim_bert[idx]:.4f}")
    
    return top_tfidf_idx, top_bert_idx

def show_comparison(tfidf_idx, bert_idx):
    
    image_names = list(all_captions.keys())
    
    plt.figure(figsize=(15,6))
    
    # TF-IDF row
    for i, idx in enumerate(tfidf_idx):
        img_path = os.path.join(IMAGES_PATH, image_names[idx])
        
        if os.path.exists(img_path):
            img = Image.open(img_path)
            
            plt.subplot(2, len(tfidf_idx), i+1)
            plt.imshow(img)
            plt.title(f"TF-IDF #{i+1}")
            plt.axis('off')
    
    # BERT row
    for i, idx in enumerate(bert_idx):
        img_path = os.path.join(IMAGES_PATH, image_names[idx])
        
        if os.path.exists(img_path):
            img = Image.open(img_path)
            
            plt.subplot(2, len(bert_idx), len(bert_idx)+i+1)
            plt.imshow(img)
            plt.title(f"BERT #{i+1}")
            plt.axis('off')
    
    plt.show()

query = "a child playing football"

tfidf_idx, bert_idx = compare_models(query)
show_comparison(tfidf_idx, bert_idx)

pip install openai-clip torch torchvision

import clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, preprocess = clip.load("ViT-B/32", device=device)

import os
import torch
from PIL import Image
from tqdm import tqdm # Highly recommended for progress bars!
clip_image_embeddings = []
# We must keep track of which images actually succeeded,
# so the final list of names matches your list of embeddings perfectly!
valid_image_names = []
# Using tqdm() will give you a nice progress bar
for img_name in tqdm(image_names, desc="Extrayant les embeddings CLIP"):
    
    img_path = os.path.join(r"C:\Users\GIGABYTE\Downloads\ImageCaptioningproject-CNN-LSTM-main (2)\Images", img_name)
    
    # 1. FIX: Check if the file actually exists on your hard drive first
    if not os.path.exists(img_path):
        continue
        
    try:
        # 2. FIX: .convert("RGB") is mandatory!
        # It guarantees that black & white or transparent images won't crash PyTorch.
        raw_image = Image.open(img_path).convert("RGB")
        
        image = preprocess(raw_image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = clip_model.encode_image(image)
        clip_image_embeddings.append(embedding.cpu().numpy()[0])
        valid_image_names.append(img_name)
        
    except Exception as e:
        print(f"Erreur avec {img_name}: {e}")
# 3. Overwrite the old list of names with the verified valid ones.
# This ensures that `clip_image_embeddings[5]` perfectly matches `image_names[5]`.
image_names = valid_image_names
print(f"\n✅ Terminé ! {len(clip_image_embeddings)} embeddings extraits avec succès.")


def find_similar_images_clip(query, top_k=5):
    text = clip.tokenize([query]).to(device)

    with torch.no_grad():
        text_embedding = clip_model.encode_text(text).cpu().numpy()

    similarities = cosine_similarity(text_embedding, clip_image_embeddings)[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for i in top_indices:
        results.append({
            "image": image_names[i],
            "score": similarities[i]
        })

    return results

query = "a happy moment during a celebration"

print("=== TF-IDF ===")
tfidf_results = find_similar_images(query)
show_results(tfidf_results)

print("=== BERT ===")
bert_results = find_similar_images_bert(query)
show_results(bert_results)


print("=== CLIP (BEST) ===")
clip_results = find_similar_images_clip(query)
clip_filenames = [item["image"] for item in clip_results]
# Pass the clean strings to the gallery
show_results(clip_filenames)


