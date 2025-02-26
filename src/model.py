import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, BatchNormalization, Conv2D,
                                     GlobalAveragePooling2D, Multiply, Lambda)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam



# Función para crear el modelo

def create_model(input_shape, num_classes):
    
    in_lay = Input(input_shape)

    base_model = ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
    for layer in base_model.layers[-10:]:
        layer.trainable = True
    
    pt_features = base_model(in_lay)
    bn_features = BatchNormalization()(pt_features)

    attn_layer = Conv2D(128, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(0.0001))(Dropout(0.5)(bn_features))
    attn_layer = Conv2D(64, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(0.0001))(attn_layer)
    attn_layer = Conv2D(32, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(0.0001))(attn_layer)
    attn_layer = Conv2D(1, kernel_size=(1,1), padding='valid', activation='sigmoid')(attn_layer)  # **Genera la máscara de atención**

    #Expandimos la máscara para igualar la profundidad de ResNet50
    pt_depth = base_model.output_shape[-1]
    up_c2_w = np.ones((1, 1, 1, pt_depth))
    up_c2 = Conv2D(pt_depth, kernel_size=(1,1), padding='same', activation='linear', use_bias=False, weights=[up_c2_w])
    up_c2.trainable = False
    attn_layer = up_c2(attn_layer)

    mask_features = Multiply()([attn_layer, bn_features])

    # 🔹 **Aplicamos GlobalAveragePooling2D y reescalamos**
    gap_features = GlobalAveragePooling2D()(mask_features)
    gap_mask = GlobalAveragePooling2D()(attn_layer)
    gap = Lambda(lambda x: x[0] / x[1], name='RescaleGAP')([gap_features, gap_mask])

    # 🔹 **Dropout y Capa de Clasificación**
    gap_dr = Dropout(0.5)(gap)
    out_layer = Dense(num_classes, activation='softmax')(gap_dr)

    model = Model(inputs=[in_lay], outputs=[out_layer])

    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),  
    metrics=['accuracy']
    )


    return model


attention_model = create_model((224, 224, 3), 5)  # 5 clases
attention_model.summary()
attention_model.save("/mnt/c/Users/Usuario/Documents/DiagnosticoRetina/models/resnet50_aptos.h5")























"""

# Definir el modelo preentrenado ResNet50
model_base = tf.keras.applications.ResNet50(
        include_top=False,  
        weights='imagenet',  
        input_shape=(224, 224, 3),  
        pooling="avg"
)

# Descongelamos las últimas 20 capas para permitir aprendizaje
for layer in model_base.layers[-20:]:  
    layer.trainable = True

# Construir la parte personalizada de la red
x = model_base.output 
x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)  # 🔹 Agregamos regularización L2
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)  # 🔹 Regularización en la capa de 64 neuronas
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

outputs = Dense(5, activation='softmax')(x)

# Modelo final
model = tf.keras.models.Model(inputs=model_base.input, outputs=outputs)

# Compilar el modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
    metrics=['accuracy']
)

# Guardar modelo inicial
model.save("/mnt/c/Users/Usuario/Documents/DiagnosticoRetina/models/resnet50_aptos.h5")


"""


