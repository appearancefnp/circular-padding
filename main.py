import tensorflow as tf

from unet import unet


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
loss = tf.keras.losses.BinaryCrossentropy()
metric = tf.keras.metrics.Accuracy()

# https://github.com/minoring/unet/blob/643b7b91aba7ec8d251b42ac79cfb07a1add0205/data.py#L16
