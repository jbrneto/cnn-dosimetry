import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv3D, BatchNormalization, MaxPooling3D, Input, UpSampling3D, Conv3DTranspose, Dropout, ReLU, LeakyReLU, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Add, Activation, LayerNormalization, Dropout

import numpy as np
import keras
from keras import backend
from keras import initializers
#from keras import ops
#from keras.src.api_export import keras_export
from keras.layers import Layer

#one = tf.constant([1.0])
#euler = tf.constant([-2.718281])
@tf.function
def custom_activation(x):
    return tf.where(x > 0, tf.math.pow(10000.0*x, 2), 0)
    #return tf.where(x > 0, (1.0 / (1.0 + tf.math.pow(x, -2.718281))), 0)
    #return one / (one + tf.math.pow(x, euler))
    #return tf.math.log_sigmoid(x)

def Block(n_filters, inputs, k_size=3, max_pooling=True, layer=0, first=False, modulated=True, styled=False, decoding=False):
  name = 'Dec_'+ str(n_filters)+'_' if decoding else 'Enc_'+ str(n_filters)+'_'
  if styled:
      name = 'Sty_'+ str(n_filters)+'_'
  if layer > 0:
    name = name + '_' + str(layer) + '_'

  conv = inputs
  #if first:
  #    conv = BatchModulation()(conv)
  #conv = BatchModulation()(conv)

  conv = Conv2D(n_filters, kernel_size=k_size, strides=1, padding='same', name=name+'Conv_1')(conv)
  conv = BatchNormalization(momentum=0.1, name=name+'BN_1')(conv)#, center=False, scale=False)(conv)
  conv = ReLU(name=name+'ReLU_1')(conv)
  #conv = Dropout(0.5)(conv)
  #conv = Activation(custom_activation)(conv)
  #if modulated:
  #conv = BatchModulation()(conv)
  #conv = LayerNormalization(epsilon=1e-05)(conv)
  #bm_res = Add(bm_res, res)
  #conv = BatchNormalization(epsilon=1e-05, momentum=0.1)(conv)
  #if modulated:
  #    conv = BatchModulation()(conv)
  #conv = Conv2D(n_filters, kernel_size=k_size, strides=1, padding='same', name=name+'Conv_2')(conv)
  #conv = BatchNormalization(momentum=0.01, name=name+'BN_2')(conv)
  #conv = ReLU(name=name+'ReLU_2')(conv)
  #conv = Activation(custom_activation)(conv)
  #if modulated:
  #conv = BatchModulation()(conv)
  #conv = BatchNormalization(epsilon=1e-05, momentum=0.1, center=False, scale=False)(conv)
  #conv = LayerNormalization(epsilon=1e-05)(conv)
  #bm_res = Add(bm_res, res)
  #conv = BatchNormalization(epsilon=1e-05, momentum=0.1)(conv)
  #conv = Conv2D(n_filters, kernel_size=k_size, strides=1, padding='same', activation='relu', name=name+'Conv_3')(conv)
  #conv = BatchNormalization(epsilon=1e-05, momentum=0.1)(conv)
  #conv = ReLU()(conv)

  pool = None
  if max_pooling:
    pool = MaxPooling2D(pool_size=2, strides=2, padding='same', name=name+'Max')(conv)

  return conv, pool

def DecoderBlock(n_filters, inputs, skip_conenction, k_size=3, layer=0, styled=False):
  name = 'Dec_'+str(n_filters)
  if styled:
      name = 'Sty_'+str(n_filters)
  if layer > 0:
    name = name + '_' + str(layer) + '_'
  conv = Conv2DTranspose(n_filters, kernel_size=k_size-1, strides=2, padding='same', name=name+'Tran')(inputs)
  #if not styled:
  conv = tf.concat([conv, skip_conenction], axis=-1, name=name+'Concat')
  conv, _ = Block(n_filters, conv, k_size=k_size, max_pooling=False, layer=layer, styled=styled, decoding=True)
  return conv

def Unet(input_shape, styled=False):
  inputs = Input(input_shape, batch_size=32)

  # Encoding
  #inp_mod = BatchModulation()(inputs)
  #mod, bm = inp_mod(inputs)

  conv1, pool1 = Block(8, inputs, first=True, modulated=True) # 32
  conv2, pool2 = Block(16, pool1, modulated=True)
  conv3, pool3 = Block(32, pool2, modulated=True)
  conv4, pool4 = Block(64, pool3, modulated=True)
  conv5, _ = Block(128, pool4, max_pooling=False, modulated=True)
  # Decoding
  uConv4 = DecoderBlock(64, conv5, conv4)
  uConv3 = DecoderBlock(32, conv4, conv3)
  uConv2 = DecoderBlock(16, uConv3, conv2)
  uConv1 = DecoderBlock(8, uConv2, conv1)
  conv = Conv2D(1, activation='sigmoid', kernel_size=1, strides=1, padding='same', name="Top_Conv")(uConv1)

  if not styled:
      model = Model(inputs=inputs, outputs=conv, name='UNet')
      return model

  # Styling
  #mod = BatchModulation()(conv5)
  sConv4 = DecoderBlock(254, conv5, uConv4, styled=True)
  sConv3 = DecoderBlock(128, sConv4, uConv3, styled=True)
  sConv2 = DecoderBlock(64, sConv3, uConv2, styled=True)
  sConv1 = DecoderBlock(32, sConv2, uConv1, styled=True)
  sconv = Conv2D(3, activation='relu', kernel_size=1, strides=1, padding='same', name="Top_Style")(sConv1)

  fConv4 = DecoderBlock(254, conv5, sConv4)
  fConv3 = DecoderBlock(128, fConv4, sConv3)
  fConv2 = DecoderBlock(64, fConv3, sConv2)
  fConv1 = DecoderBlock(32, fConv2, sConv1)
  fconv = Conv2D(1, activation='sigmoid', kernel_size=1, strides=1, padding='same', name="Top_Conv")(fConv1)

  model = Model(inputs=inputs, outputs=conv, name='UNet')
  #encode = Model(inputs=inputs, outputs=conv5, name='EncodeNet')
  #decode = Model(inputs=conv5, outputs=conv, name='DecodeNet')
  smodel = Model(inputs=inputs, outputs=sconv, name='StyleNet')
  fmodel = Model(inputs=inputs, outputs=fconv, name='FinalNet')

  return model, smodel, fmodel
  #return encode, decode, smodel

#@keras_export("keras.layers.BatchModulation")
class BatchModulation(Layer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.momentum = float(0.1)#float(0.99)
    self.epsilon = float(1e-07)
    self.learning_rate = float(0.00001)

    self.forged=False
    self.hitdebug=False
    self.total_count=0
    
    self.style_idx = None
    self.style_means = []
    self.style_vars = []

    self.last_mean = None
    self.last_var = None

  def build(self, input_shape):
    self.total = input_shape[-1]
    self.hits = self.add_weight(shape=(1,),name="hits",trainable=False, initializer=initializers.get("zeros"))
    self.miss = self.add_weight(shape=(1,),name="miss",trainable=False, initializer=initializers.get("zeros"))

    xape = input_shape[-1:]
    #xape = tuple((input_shape[0], input_shape[-1]))#input_shape[-1:]
    #print(xape, ' - ', input_shape)

    self.moving_mean = self.add_weight(shape=xape,name="moving_mean",trainable=False, initializer=initializers.get("zeros"))
    self.moving_variance = self.add_weight(shape=xape, name="moving_variance", trainable=False, initializer=initializers.get("ones"))

    #self.total_mean = self.add_weight(shape=xape,name="total_mean",trainable=False, initializer=initializers.get("zeros"))
    #self.total_variance = self.add_weight(shape=xape, name="total_variance", trainable=False, initializer=initializers.get("ones"))
      
    #self.track_mean = self.add_weight(shape=xape,name="track_mean",trainable=False, initializer=initializers.get("zeros"))
    #self.track_variance = self.add_weight(shape=xape, name="track_variance", trainable=False, initializer=initializers.get("ones"))
      
    self.norm_axes = tuple(list(range(len(input_shape)))[:-1]) # batch
    #self.norm_axes = tuple(list(range(len(input_shape)))[1:]) # layer
    #self.norm_axes = tuple(list(range(len(input_shape)))[1:-1]) # instance
    self.built = True

  def set_mean_var(self, mean_var):
    self.moving_mean = mean_var[0]
    self.moving_variance = mean_var[1]
      
  def new_style(self):
    self.style_means.append(self.track_mean.value())
    self.style_vars.append(self.track_variance.value())
    #self.style_idx = len(self.style_vars)-1
    #self.total_mean.assign((self.total_mean * 0.5) + (self.track_mean.value() * (1.0 - 0.5)))
    #self.total_variance.assign((self.total_variance * 0.5) + (self.track_variance.value() * (1.0 - 0.5)))

  def use_style(self, index=None):
      if index is None:
          self.style_idx = len(self.style_vars)-1
      else:
          self.style_idx = index

  def zerocounts(self):
    self.hits.assign([0.0])# = self.add_weight(shape=(1,),name="hits",trainable=False, initializer=initializers.get("zeros"))
    self.miss.assign([0.0])# = self.add_weight(shape=(1,),name="miss",trainable=False, initializer=initializers.get("zeros"))
  def hitdebugOn(self):
    self.hitdebug=True
  def hitdebugOff(self):
    self.hitdebug=False
      
  def forge(self):
    self.forged=True
    
  def call(self, inputs, training=False):
    mean, variance = self._normalize(inputs)
    #self.last_mean = mean
    #self.last_var = variance
    if training:
        if not self.forged:
            self.moving_mean.assign((self.moving_mean * self.momentum) + (mean * (1.0 - self.momentum)))
            self.moving_variance.assign((self.moving_variance * self.momentum) + (variance * (1.0 - self.momentum)))
    #else:
        
    return inputs
    #if self.forged:
    #mean, variance = self._normalize(inputs)
    #if training:
    #    self.moving_mean.assign((self.moving_mean * self.momentum) + (mean * (1.0 - self.momentum)))
    #    self.moving_variance.assign((self.moving_variance * self.momentum) + (variance * (1.0 - self.momentum)))
        
    #return tf.nn.batch_normalization(x=inputs, mean=mean, variance=variance, offset=self.moving_mean, scale=self.moving_variance, variance_epsilon=self.epsilon)
      
      
    if training:
        mean, variance = self._normalize(inputs)
        
        if not self.forged:            
            self.moving_mean.assign((self.moving_mean * self.momentum) + (mean * (1.0 - self.momentum)))
            self.moving_variance.assign((self.moving_variance * self.momentum) + (variance * (1.0 - self.momentum)))
            self.track_mean.assign((self.track_mean * self.momentum) + (mean * (1.0 - self.momentum)))
            self.track_variance.assign((self.track_variance * self.momentum) + (variance * (1.0 - self.momentum)))
            self.total_mean.assign((self.total_mean * self.momentum) + (mean * (1.0 - self.momentum)))
            self.total_variance.assign((self.total_variance * self.momentum) + (variance * (1.0 - self.momentum)))
            #self.total_mean.assign((self.total_mean * 0.99) + (mean * (1.0 - 0.99)))
            #self.total_variance.assign((self.total_variance * 0.99) + (variance * (1.0 - 0.99)))

        else:
            #if self.style_idx is None:
            man = self.track_mean.assign((self.track_mean * self.momentum) + (mean * (1.0 - self.momentum)))
            var = self.track_variance.assign((self.track_variance * self.momentum) + (variance * (1.0 - self.momentum)))
            #else:
            #    man = self.style_means[self.style_idx]
            #    var = self.style_vars[self.style_idx]

            #self.total_mean.assign((self.total_mean * 0.99) + (mean * (1.0 - 0.99)))
            #self.total_variance.assign((self.total_variance * 0.99) + (variance * (1.0 - 0.99)))
            
            #diff_m = self.track_mean - self.moving_mean
            diff_v = tf.math.sqrt(self.moving_variance / (var+self.epsilon))
            #diff_v = tf.math.sqrt(self.track_variance / (self.moving_variance+self.epsilon))
            
            #new_inputs = inputs - self.diff_m
            #new_inputs = inputs - diff_m
            #inputs = tf.where(inputs > 0., (inputs - diff_m), 0.)
            
            #mean, variance = self._normalize(inputs)
            
            #diff_v = tf.math.sqrt(self.moving_variance / (variance+self.epsilon))
            
            #new_inputs = new_mean + ((inputs - new_mean) * diff_v)
            #inputs = tf.where(inputs > 0., self.moving_mean + ((inputs - mean) * diff_v), 0.)
            #inputs = self.moving_mean + ((inputs - diff_m) * diff_v)
            inputs = self.moving_mean + ((inputs - man) * diff_v)

            #mean, variance = self._normalize(inputs)
            #self.total_mean.assign((self.total_mean * self.momentum) + (mean * (1.0 - self.momentum)))
            #self.total_variance.assign((self.total_variance * self.momentum) + (variance * (1.0 - self.momentum)))

            #self.moving_mean.assign((self.moving_mean * self.momentum) + (mean * (1.0 - self.momentum)))
            #self.moving_variance.assign((self.moving_variance * self.momentum) + (variance * (1.0 - self.momentum)))
    else:
        if self.forged:
            if self.style_idx is None:
                mean = self.track_mean
                var = self.track_variance
            elif self.style_idx == -2:
                mean = self.total_mean
                var = self.total_variance
            elif self.style_idx == -1:
                mean = tf.reduce_mean(self.style_means, axis=0)
                var = tf.reduce_mean(self.style_vars, axis=0)
            else:
                mean = self.style_means[self.style_idx]
                var = self.style_vars[self.style_idx]
                
            diff_v = tf.math.sqrt(self.moving_variance / (var+self.epsilon))
            #diff_v = tf.math.sqrt(var / (self.moving_variance+self.epsilon))
            inputs = self.moving_mean + ((inputs - mean) * diff_v)
            
    return inputs
    
  def _normalize(self, x, ignore=None, new_axis=None):
    ax = self.norm_axes
    if new_axis is not None:
        ax = new_axis
    return tf.nn.moments(x, ax)#, keepdims=True)

  def get_config(self):
    base_config = super().get_config()
    config = {}
    return {**base_config, **config}
