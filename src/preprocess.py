import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# RUTAS PARA EL DATASET
DATA_DIR = "/content/drive/MyDrive/DiagnosticoRetina/data/APTOS2019/train_images"
LABELS_FILE = "/content/drive/MyDrive/DiagnosticoRetina/data/APTOS2019/train.csv"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Cargar CSV con etiquetas
df = pd.read_csv(LABELS_FILE)
df['image_path'] = df['id_code'].apply(lambda x: os.path.join(DATA_DIR, x + ".png"))

# Separar en entrenamiento y validación
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["diagnosis"])


# Función de aumento de datos (Data Augmentation)
def augment_image(image):
    image = tf.image.random_flip_left_right(image)  
    image = tf.image.random_flip_up_down(image)  
    image = tf.image.random_brightness(image, max_delta=0.2)  
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  
    return image

# Función para crear datasets 
def parse_image(filename, label, augment=False):

    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0  #Normalizar 
    
    if not augment:
        return image, label

    def do_augment():
        return augment_image(image)

    def do_not_augment():
        return image

    # 4. Evaluar condición: se augmenta solo si label es 2, 3 o 4
    condition = tf.logical_or(
                    tf.equal(label, 2),
                    tf.logical_or(
                        tf.equal(label, 3),
                        tf.equal(label, 4)
                    )
                )
    image = tf.cond(condition, do_augment, do_not_augment)
    return image, label

def load_dataset(file_paths, labels, batch_size=32, augment=False, shuffle=True):
    # Aquí usamos un lambda para pasar el flag augment a parse_image
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(lambda f, l: parse_image(f, l, augment),
                          num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(len(file_paths))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Creamos los datasets
train_file_paths = train_df['image_path'].values
train_labels = train_df['diagnosis'].values.astype(np.int32)
train_dataset = load_dataset(train_file_paths, train_labels, batch_size=BATCH_SIZE, augment=True)

val_file_paths = val_df['image_path'].values
val_labels = val_df['diagnosis'].values.astype(np.int32)
val_dataset = load_dataset(val_file_paths, val_labels, batch_size=BATCH_SIZE, augment=False)

