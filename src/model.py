import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.regularizers import l2


# Definir el modelo preentrenado ResNet50
model_base = tf.keras.applications.ResNet50(
        include_top=False,  
        weights='imagenet',  
        input_shape=(224, 224, 3),  
        pooling="avg"
)

# Inicialmente, congelamos todas las capas del modelo base
model_base.trainable = False 

# Construir la parte personalizada de la red
x = model_base.output 
x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)  # 🔹 Agregamos regularización L2
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)  # 🔹 Regularización en la capa de 64 neuronas
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

outputs = Dense(5, activation='softmax')(x)

# Modelo final
model = tf.keras.models.Model(inputs=model_base.input, outputs=outputs)

# Compilar el modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.1),  # 🔹 Usamos label smoothing
    metrics=['accuracy']
)

# Guardar modelo inicial
model.save("/mnt/c/Users/Usuario/Documents/DiagnosticoRetina/models/resnet50_aptos.h5")


