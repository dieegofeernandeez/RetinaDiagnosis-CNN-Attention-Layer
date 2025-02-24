import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tensorflow as tf
from src.preprocess import train_dataset, val_dataset
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping


# Importar el modelo cargado 
model=tf.keras.models.load_model('/mnt/c/Users/Usuario/Documents/DiagnosticoRetina/models/resnet50_aptos.h5')

# Definir los hiperparámetros

num_epochs=10

checkpoint_cb = ModelCheckpoint(
    "/mnt/c/Users/Usuario/Documents/DiagnosticoRetina/models/resnet50_aptos_best.h5",  # 📌 Ruta donde guardamos el modelo
    save_best_only=True,  # ✅ Solo guarda el mejor modelo (según val_loss)
    monitor="val_loss",  # 🔍 Monitorea la pérdida en validación
    mode="min",  # 🔽 Queremos que val_loss sea lo más baja posible
    verbose=1  # 📝 Muestra información en la consola
)


early_stopping_cb = EarlyStopping(
    patience=4,  # 📌 Si no mejora en 3 epochs, se detiene el entrenamiento
    restore_best_weights=True,  # ✅ Restaura los pesos del mejor modelo antes de la caída
    monitor="val_loss",  # 🔍 Monitorea la pérdida en validación
    mode="min",  # 🔽 Queremos que val_loss sea lo más baja posible
    verbose=1  # 📝 Muestra información en la consola
)



# Entrenar el modelo
history = model.fit(
    train_dataset,  
    validation_data=val_dataset,  
    epochs=num_epochs,  
    callbacks=[checkpoint_cb, early_stopping_cb],  # Callbacks para guardar el mejor modelo y detener entrenamiento temprano
    verbose=1  # Muestra el progreso en la consola
)
