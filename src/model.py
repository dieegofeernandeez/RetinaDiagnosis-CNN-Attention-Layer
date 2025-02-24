import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.applications.resnet50 import preprocess_input


# Definir el modelo preentrenado que se va utilizar en este caso ResNet50
model_base = tf.keras.applications.ResNet50(
        include_top=False,  #  No incluir la capa de clasificación final 
        weights='imagenet',  #  Usar pesos preentrenados en ImageNet
        input_shape=(224, 224, 3),  # Tamaño de entrada de las imagenes 
        pooling="avg"
)

model_base.trainable = False #congelar las capas para que  no se entrenen y evitar perder el conocimiento


# Añadir las capas de clasificación para nuestro problema
x = model_base.output 
x = Dense(128, activation = 'relu')(x)
x = Dropout(0.3)(x)
x = Dense(64, activation = 'relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(5, activation = 'softmax')(x)


# Modelo final para conectar la RestNet50 con nuestras capas densas
model = tf.keras.models.Model(inputs=model_base.input, outputs=outputs)


#Compilamos
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.save("/mnt/c/Users/Usuario/Documents/DiagnosticoRetina/models/resnet50_aptos.h5")

