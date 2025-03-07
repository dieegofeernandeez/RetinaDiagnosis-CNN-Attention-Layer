import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, BatchNormalization, Conv2D,
                                     GlobalAveragePooling2D, Multiply, Lambda)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from customlayers import rescale_gap


# Función para crear el modelo

def create_model(input_shape, num_classes):
    
    in_lay = Input(input_shape)
    base_model = ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    
    pt_features = base_model(in_lay)
    bn_features = BatchNormalization()(pt_features)

    attn_layer = Conv2D(128, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(0.0005))(Dropout(0.6)(bn_features))
    attn_layer = Conv2D(64, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(0.0005))(attn_layer)
    attn_layer = Conv2D(32, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(0.0005))(attn_layer)
    attn_layer = Conv2D(1, kernel_size=(1,1), padding='valid', activation='sigmoid')(attn_layer)  # **Genera la máscara de atención**

    #Expandimos la máscara para igualar la profundidad de ResNet50
    pt_depth = base_model.output_shape[-1]
    up_c2_w = np.ones((1, 1, 1, pt_depth))
    up_c2 = Conv2D(
        pt_depth,
        kernel_size=(1,1),
        padding='same',
        activation='linear',
        use_bias=False
    )
    _ = up_c2(tf.zeros((1, 7, 7, 1)))
    up_c2.set_weights([up_c2_w])
    up_c2.trainable = False
    attn_layer = up_c2(attn_layer)
    mask_features = Multiply()([attn_layer, bn_features])
    
    gap_features = GlobalAveragePooling2D()(mask_features)
    gap_mask = GlobalAveragePooling2D()(attn_layer)
    gap = Lambda(rescale_gap, name='RescaleGAP')([gap_features, gap_mask])

    gap_dr = Dropout(0.6)(gap)
    out_layer = Dense(num_classes, activation='softmax')(gap_dr)

    model = Model(inputs=[in_lay], outputs=[out_layer])

    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),  
    metrics=['accuracy']
    )

    return model


attention_model = create_model((224, 224, 3), 5)  
attention_model.summary()
attention_model.save("/content/drive/MyDrive/DiagnosticoRetina/models/resnet50_aptos.keras")