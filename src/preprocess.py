import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split


# RUTAS PARA EL DATASET

DATA_DIR = "/mnt/c/Users/Usuario/Documents/DiagnosticoRetina/data/APTOS2019/train_images"
LABELS_FILE = "/mnt/c/Users/Usuario/Documents/DiagnosticoRetina/data/APTOS2019/train.csv"
IMAGE_SIZE = (224, 224)

df = pd.read_csv(LABELS_FILE)
df['image_path'] = df['id_code'].apply(lambda x: os.path.join(DATA_DIR, x + ".png")) # Creamos una nueva columna que contiene la ruta de cada imagen en /train_images

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["diagnosis"])

# Función para cargar y preprocesar las imágenes

def load_and_preprocess_image(img_path):
    img = cv2.imread(img_path.numpy().decode("utf-8"))  # Leer imagen
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
    img = cv2.resize(img, IMAGE_SIZE)  # Redimensionar a 224x224
    img = img / 255.0  # Normalizar valores (0-1)
    return img.astype(np.float32)

# Creamos la funcíon para el dataset

def create_tf_dataset(df, batch_size=32):
    image_paths = df['image_path'].values
    labels = df['diagnosis'].values  # Las etiquetas de clasificación
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda x, y: (tf.py_function(load_and_preprocess_image, [x], tf.float32), y))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset


train_dataset = create_tf_dataset(train_df)
val_dataset = create_tf_dataset(val_df)

