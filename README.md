# Diagnóstico de Retinopatía con CNN + Capa de atención

Este proyecto implementa un modelo de red neuronal convolucional (CNN) basado en ResNet50 con una capa de atención personalizada para el diagnóstico automático de **Retinopatía Diabética** a partir de imágenes de retina.

El dataset utilizado es el **APTOS 2019**, disponible en Kaggle:  
https://www.kaggle.com/competitions/aptos2019-blindness-detection/data

Contiene imágenes del fondo del ojo anotadas con 5 clases de diagnóstico:

- 0 — No DR
- 1 — DR leve
- 2 — DR moderada
- 3 — DR severa
- 4 — DR proliferativa

## Objetivo

Desarrollar un modelo capaz de clasificar imágenes de retina en las 5 clases anteriores, mejorando la atención del modelo mediante una **capa de atención personalizada** aplicada sobre la salida de ResNet50.

## Entrenamiento

### Requisitos

Instala las dependencias desde `requirements.txt`:
'bash'
pip install -r requirements.txt


### Preprocesamiento

Las imágenes se cargan, redimensionan y normalizan. Se aplican aumentos de datos condicionales (solo a clases 2, 3 y 4).

Esto se maneja en src/preprocess.py.

### Arquitectura

- ResNet50 (preentrenada con ImageNet, sin top)

- Capa de atención convolucional sobre las características profundas

- Mecanismo GAP con ajuste personalizado (RescaleGAP)

- Capa densa final con softmax

Definido en src/model.py.


### Entrenamiento

Para entrenar el modelo:

'bash'
python train.py

El modelo se guarda automáticamente en models/ y se monitoriza usando:

- ModelCheckpoint

- EarlyStopping

- ReduceLROnPlateau

### Notas finales




