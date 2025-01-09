import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv3D, BatchNormalization, MaxPooling3D, Input, UpSampling3D, Conv3DTranspose, Dropout, ReLU, LeakyReLU, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose

#Custom model: https://www.tensorflow.org/tutorials/customization/custom_layers#implementing_custom_layers
#Data augmentation: https://www.tensorflow.org/tutorials/images/classification#data_augmentation
#https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/
#Unet: https://www.tensorflow.org/tutorials/images/segmentation

#https://stackoverflow.com/questions/49295311/what-is-the-difference-between-flatten-and-globalaveragepooling2d-in-keras
#If youre overfitting, you might want to try GlobalAveragePooling2D

def Block(n_filters, inputs, max_pooling=True):
  name = 'Enc_'+ str(n_filters)+'_' if max_pooling else 'Dec_'+ str(n_filters)+'_'
  conv = Conv3D(n_filters, kernel_size=3, strides=1, padding='same', activation='relu', name=name+'Conv_1')(inputs)
  conv = Conv3D(n_filters, kernel_size=3, strides=1, padding='same', activation='relu', name=name+'Conv_2')(conv)
  #conv = BatchNormalization(epsilon=1e-05, momentum=0.1)(conv)
  #conv = LeakyReLU(alpha=0.2)(conv)
  #conv = ReLU()(conv)
  #conv = Dropout(rate=0.5)(conv)

  pool = None
  if max_pooling:
    pool = MaxPooling3D(pool_size=2, strides=2, padding='same', name=name+'Max')(conv)

  return conv, pool

def DecoderBlock(n_filters, inputs, skip_conenction):
  name = 'Dec_'+str(n_filters)
  conv = Conv3DTranspose(n_filters, kernel_size=2, strides=2, padding='same', name=name+'Tran')(inputs)
  #conv = tf.concat([conv, skip_conenction], axis=4, name=name+'Concat')
  conv = tf.concat([conv, skip_conenction], axis=-1, name=name+'Concat')
  conv, _ = Block(n_filters, conv, max_pooling=False)
  return conv

def Unet(input_shape):
  inputs = Input(input_shape)

  # Encoding
  conv1, pool1 = Block(16, inputs) #32
  conv2, pool2 = Block(32, pool1) #64
  conv3, pool3 = Block(64, pool2) #128
  conv4, pool4 = Block(128, pool3) #256
  conv5, _     = Block(256, pool4, max_pooling=False) #512
  # Decoding
  uConv4 = DecoderBlock(128, conv5, conv4) #256
  uConv3 = DecoderBlock(64, uConv4, conv3) #128
  uConv2 = DecoderBlock(32, uConv3, conv2) #64
  uConv1 = DecoderBlock(16, uConv2, conv1) #32
  #uConv1 = DecoderBlock(32, pool1, conv1)
  
  # nao tem flatten

  conv = Conv3D(1, activation='sigmoid', kernel_size=1, strides=1, padding='same', name="Top_Conv")(uConv1)

  model = Model(inputs=inputs, outputs=conv, name='UNet')

  return model