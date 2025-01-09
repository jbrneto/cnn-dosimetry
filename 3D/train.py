import tensorflow as tf
import numpy as np
from data import InstanceLoader
from model import Unet

instance_dir = 'C:/Users/joao/Desktop/tese/datasets/Pancreatic/Pancreatic-Preprocessed'

instance_loader = InstanceLoader(instance_dir,batch_size=1)
instance_shape = instance_loader.shape()
print(instance_shape)

unet = Unet(instance_shape)
unet.compile(
	optimizer=tf.keras.optimizers.Adam(), 
	loss=tf.keras.losses.MeanSquaredError(), 
	metrics=['accuracy']
)

print(unet.summary())

results = unet.fit(instance_loader, batch_size=1, epochs=5)