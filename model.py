import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv3D, BatchNormalization, MaxPooling3D, Input, UpSampling3D, Dropout, ReLU, LeakyReLU


def Block(n_filters, inputs, max_pooling=True):
  conv = Conv3D(n_filters, kernel_size=3, strides=1, padding='same')(inputs)
  #conv = BatchNormalization(epsilon=1e-05, momentum=0.1)(conv)
  #conv = LeakyReLU(alpha=0.2)(conv)
  conv = ReLU()(conv)
  #conv = Dropout(rate=0.5)(conv)

  skip_conenction = conv
  if max_pooling:
      conv = MaxPooling3D(pool_size=2, strides=2, padding='same')(conv)
  return conv, skip_conenction

def DecoderBlock(n_filters, inputs, skip_conenction):
  conv = UpSampling3D(size=2)(inputs)
  conv = tf.concat([conv, skip_conenction], axis=4)
  conv = Block(n_filters, conv, max_pooling=False)
  return conv[0]

#def Block(n_filters, inputs, max_pooling=True):
#  conv = Conv3D(n_filters, activation='relu', kernel_size=3, strides=1, padding='same')(inputs)
#  conv = BatchNormalization(epsilon=1e-05, momentum=0.1)(conv)
#  conv = Conv3D(n_filters, activation='relu', kernel_size=3, strides=1, padding='same')(conv)
#  conv = BatchNormalization(epsilon=1e-05, momentum=0.1)(conv)
#  skip_conenction = conv
#  if max_pooling:
#      conv = MaxPooling3D(pool_size=2, strides=2, padding='same')(conv)
#  return conv, skip_conenction

#def DecoderBlock(n_filters, inputs, skip_conenction):
#  conv = UpSampling3D(size=2)(inputs)
#  conv = Conv3D(n_filters, kernel_size=3, strides=1, padding='same')(conv)
#  conv = tf.concat([conv, skip_conenction], axis=4)
#  conv = Block(n_filters, conv, max_pooling=False)
#  return conv[0]

def Unet(input_shape):
  inputs = Input(input_shape)

  # Encoding
  e1 = Block(32, inputs, max_pooling=True)
  #e2 = Block(64, e1[0], max_pooling=True)
  #e3 = Block(128, e2[0], max_pooling=True)
  #e4 = Block(256, e3[0], max_pooling=True)
  #e5 = Block(512, e4[0], max_pooling=False)
  # Decoding
  #d4 = DecoderBlock(256, e5[0], e4[1])
  #d3 = DecoderBlock(128, d4, e3[1])
  #d2 = DecoderBlock(64, d3, e2[1])
  #d1 = DecoderBlock(32, d2, e1[1])
  #d3 = DecoderBlock(128, e3[0], e3[1])
  #d2 = DecoderBlock(64, e2[0], e2[1])
  d1 = DecoderBlock(32, e1[0], e1[1])
  
  conv = Conv3D(1, activation='relu', kernel_size=1, strides=1, padding='same')(d1)

  model = Model(inputs=inputs, outputs=conv)

  return model