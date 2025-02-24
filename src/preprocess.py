import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# RUTAS PARA EL DATASET

DATA_DIR = "/mnt/c/Users/Usuario/Documents/DiagnosticoRetina/data/APTOS2019/train_images"
LABELS_FILE = "/mnt/c/Users/Usuario/Documents/DiagnosticoRetina/data/APTOS2019/train.csv"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Cargar CSV con etiquetas
df = pd.read_csv(LABELS_FILE)
df['image_path'] = df['id_code'].apply(lambda x: os.path.join(DATA_DIR, x + ".png"))

# Separar en entrenamiento y validación (estratificado)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["diagnosis"])

# 🔹 Función de aumento de datos (Data Augmentation)
def augment_image(image):
    image = tf.image.random_flip_left_right(image)  # Voltear horizontalmente
    image = tf.image.random_flip_up_down(image)  # Voltear verticalmente
    image = tf.image.random_brightness(image, max_delta=0.2)  # Cambios en brillo
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # Cambios en contraste
    return image

# 🔹 Función para cargar y preprocesar imágenes
def load_and_preprocess_image(img_path, label, augment=False):
    img = cv2.imread(img_path.numpy().decode("utf-8"))  # Cargar imagen
    if img is None:
        print(f"⚠️ Imagen no encontrada: {img_path.numpy().decode('utf-8')}")
        return np.zeros((*IMAGE_SIZE, 3), dtype=np.float32), label  # Evita fallos

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)

    img = img.astype(np.float32) / 127.5 - 1.0  # 🔹 Convertimos correctamente a float32 en [-1,1]

    if augment:
        img = augment_image(tf.convert_to_tensor(img))  # 🔹 Convertimos a tensor para usar augmentaciones

    return img, label

# 🔹 Función para crear datasets en formato TensorFlow
def create_tf_dataset(df, batch_size=32, augment=False):
    image_paths = df['image_path'].values
    labels = df['diagnosis'].values.astype(np.int32) 
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda x, y: tf.py_function(
        func=load_and_preprocess_image, 
        inp=[x, y, augment], 
        Tout=(tf.float32, tf.int32)
    ), num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Crear datasets de entrenamiento y validación
train_dataset = create_tf_dataset(train_df, batch_size=BATCH_SIZE, augment=True)
val_dataset = create_tf_dataset(val_df, batch_size=BATCH_SIZE, augment=False)

