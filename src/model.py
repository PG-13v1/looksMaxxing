import tensorflow as tf
from tensorflow.keras import layers, Model


def build_model(config, tab_dim=None):
    use_img = config['FEATURES']['use_image_features']
    use_tab = config['FEATURES']['use_landmark_features']
    img_in, tab_in = None, None

    if use_img:
        img_shape = (*tuple(config['DATA']['image_size']), 3)
        img_in = layers.Input(shape=img_shape, name='image_input')
        base = tf.keras.applications.MobileNetV2(include_top=False, pooling='avg')(img_in)
        x = layers.Dense(256, activation='relu')(base)
        x = layers.Dropout(0.3)(x)

    if use_tab:
        if tab_dim is None:
            raise ValueError("tab_dim must be provided when use_landmark_features is True")
        tab_in = layers.Input(shape=(tab_dim,), name='tab_input')
        t = layers.Dense(128, activation='relu')(tab_in)
        t = layers.Dropout(0.3)(t)

    if use_img and use_tab:
        z = layers.Concatenate()([x, t])
    elif use_img:
        z = x
    else:
        z = t

    z = layers.Dense(64, activation='relu')(z)
    out = layers.Dense(1, activation='linear')(z)

    return Model(inputs=[i for i in [img_in, tab_in] if i is not None], outputs=out)