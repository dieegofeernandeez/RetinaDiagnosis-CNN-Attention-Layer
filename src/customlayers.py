import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="Custom", name="rescale_gap")

def rescale_gap(inputs, epsilon=1e-8):
    gap_features, gap_mask = inputs
    return gap_features / (gap_mask + epsilon)

