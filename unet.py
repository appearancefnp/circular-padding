import tensorflow as tf


def unet(flags_obj, n_filters=64):

    # Contracting Path (encoding)
    inputs = tf.keras.layers.Input(flags_obj.input_size)
    conv1 = tf.keras.layers.Conv2D(n_filters,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer='he_normal')(inputs)
    conv1 = tf.keras.layers.Conv2D(n_filters,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer='he_normal')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(n_filters * 2,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer='he_normal')(pool1)
    conv2 = tf.keras.layers.Conv2D(n_filters * 2,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer='he_normal')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(n_filters * 4,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer='he_normal')(pool2)
    conv3 = tf.keras.layers.Conv2D(n_filters * 4,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer='he_normal')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = tf.keras.layers.Conv2D(n_filters * 8,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer='he_normal')(pool3)
    conv4 = tf.keras.layers.Conv2D(n_filters * 8,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer='he_normal')(conv4)
    drop4 = tf.keras.layers.Dropout(0.3)(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = tf.keras.layers.Conv2D(n_filters * 16,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer='he_normal')(pool4)
    conv5 = tf.keras.layers.Conv2D(n_filters * 16,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer='he_normal')(conv5)
    drop5 = tf.keras.layers.Dropout(0.3)(conv5)

    # Expansive Path (decoding)
    up6 = tf.keras.layers.Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2),
                                          padding='same')(drop5)
    merge6 = tf.keras.layers.concatenate([up6, drop4], axis=3)
    conv6 = tf.keras.layers.Conv2D(n_filters * 8,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer='he_normal')(merge6)
    conv6 = tf.keras.layers.Conv2D(n_filters * 8,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer='he_normal')(conv6)

    up7 = tf.keras.layers.Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2),
                                          padding='same')(conv6)
    merge7 = tf.keras.layers.concatenate([up7, conv3], axis=3)
    conv7 = tf.keras.layers.Conv2D(n_filters * 4,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer='he_normal')(merge7)
    conv7 = tf.keras.layers.Conv2D(n_filters * 4,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer='he_normal')(conv7)

    up8 = tf.keras.layers.Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2),
                                          padding='same')(conv7)
    merge8 = tf.keras.layers.concatenate([up8, conv2], axis=3)
    conv8 = tf.keras.layers.Conv2D(n_filters * 2,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer='he_normal')(merge8)
    conv8 = tf.keras.layers.Conv2D(n_filters * 2,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer='he_normal')(conv8)

    up9 = tf.keras.layers.Conv2DTranspose(n_filters, (3, 3), strides=(2, 2),
                                          padding='same')(conv8)
    merge9 = tf.keras.layers.concatenate([up9, conv1], axis=3)
    conv9 = tf.keras.layers.Conv2D(n_filters,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer='he_normal')(merge9)
    conv9 = tf.keras.layers.Conv2D(n_filters,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer='he_normal')(conv9)
    conv9 = tf.keras.layers.Conv2D(2,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer='he_normal')(conv9)

    conv10 = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model
