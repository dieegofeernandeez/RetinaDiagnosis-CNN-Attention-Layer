import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tensorflow as tf
from src.preprocess import train_dataset, val_dataset
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ReduceLROnPlateau


# Importar el modelo cargado 
model=tf.keras.models.load_model('/mnt/c/Users/Usuario/Documents/DiagnosticoRetina/models/resnet50_aptos.h5')

# Definir los hiperparámetros

num_epochs=25

checkpoint_cb = ModelCheckpoint(
    "/mnt/c/Users/Usuario/Documents/DiagnosticoRetina/models/resnet50_aptos_best.h5",  # 📌 Ruta donde guardamos el modelo
    save_best_only=True,  # ✅ Solo guarda el mejor modelo (según val_loss)
    monitor="val_loss",  # 🔍 Monitorea la pérdida en validación
    mode="min",  # 🔽 Queremos que val_loss sea lo más baja posible
    verbose=1  # 📝 Muestra información en la consola
)


early_stopping_cb = EarlyStopping(
    patience=3,  # 📌 Si no mejora en 3 epochs, se detiene el entrenamiento
    restore_best_weights=True,  # ✅ Restaura los pesos del mejor modelo antes de la caída
    monitor="val_loss",  # 🔍 Monitorea la pérdida en validación
    mode="min",  # 🔽 Queremos que val_loss sea lo más baja posible
    verbose=1  # 📝 Muestra información en la consola
)

lr_scheduler = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,  # Reduce el LR a la mitad si no mejora
    patience=2,  # Espera 2 epochs sin mejora antes de reducir
    verbose=1
)



LABELS_FILE = "/mnt/c/Users/Usuario/Documents/DiagnosticoRetina/data/APTOS2019/train.csv"  # Ruta al CSV

# Cargar etiquetas
df = pd.read_csv(LABELS_FILE)
train_labels = df["diagnosis"].values  # Extraer las etiquetas

# Calcular los pesos de clase
class_weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)

# Convertir los pesos a diccionario
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

# Entrenar el modelo
history = model.fit(
    train_dataset,  
    validation_data=val_dataset,  
    epochs=num_epochs,  
    class_weight=class_weights_dict,  # 🔴 **Aquí aplicamos los pesos de clase**
    callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler],  # Callbacks para guardar el mejor modelo y detener entrenamiento temprano
    verbose=1  # Muestra el progreso en la consola
)
 