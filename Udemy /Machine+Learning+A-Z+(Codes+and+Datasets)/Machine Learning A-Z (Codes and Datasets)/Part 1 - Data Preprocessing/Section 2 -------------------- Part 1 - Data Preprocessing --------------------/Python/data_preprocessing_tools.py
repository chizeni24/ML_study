# Load the Libraries
import tensorflow as tf
import numpy as np
import keras
from keras.layers import Conv2D, Conv2DTranspose, Input,Dense,Flatten,Lambda ,Reshape, BatchNormalization
from keras.models import Model
from keras.metrics import binary_crossentropy
from keras import backend as K
from keras.datasets import mnist
import matplotlib.pyplot as plt

#Load MNIST Dataset
(x_train, _),(x_test,_) = mnist.load_data()

#Normalize the data based on the max value of the X_test aand train 
x_train = x_train.astype('float32')/225   #
x_test = x_test.astype('float32')/225


img_width = x_train.shape[1]    #set the image width 
img_height = x_train.shape[2]   #set the image of the height 
Number_of_channels  = 1

x_train = x_train.reshape(x_train.shape[0],img_width, img_height, Number_of_channels) # 6000,28,28,1, define the x train matrix 
x_test = x_test.reshape(x_test.shape[0],img_width, img_height, Number_of_channels) # 6000,28,28,1, define the x train matrix

latent_dim  = 2
Input_Shape = (img_height,img_width, Number_of_channels)  # 28,28,1 for the encoder : it has to have the same shape as the image entering 

#Takes in the input and compresses, learn both the mean and variance of the data and 
E_Input = Input(shape = Input_Shape, name = 'Encoder_Model_Inputs')
# Define a few more layersInput_Shape
E = Conv2D(32,3,padding='same',activation='relu')(E_Input)
E = Conv2D(64,3,padding='same',activation='relu',strides=(2,2))(E)
E = Conv2D(128,3,padding='same',activation='relu')(E)
E = Conv2D(64,3,padding='same',activation='relu')(E)
E = Conv2D(32,3,padding='same',activation='relu')(E_Input) #preflatten layer

######################################################intermidiary step #################################################################
Initial_Shape = K.int_shape(E)     #some backend imporatation to activate the keras:  to save the preflatten layer shape for unzipping
#########################################################################################################################################
E = Flatten()(E)    #flattten information 
E = Dense(32,activation='relu')(E) #compression layer that is dependent on the dimension of preflatten layer last learning phase 
mean = Dense(latent_dim, name ='mean')(E)  #learn the mean 
variance = Dense(latent_dim, name='variance')(E)  #learn  variance of the data

#Create a function to enabling mapping from the latent space
def Sample_Latent_Space(mean_variance):

  mean, variance = mean_variance
  eps = K.random_normal(shape = (K.shape(mean)[0], K.int_shape(mean)[1]))  #noise
  Mapped_Space = mean + K.exp(variance/2)* eps
  return Mapped_Space



Mapped_Space = Lambda(Sample_Latent_Space, output_shape = (latent_dim,), name = 'MS')([mean,variance])
encoder = Model(E_Input, [mean, variance, Mapped_Space])

D_Input = Input(shape= (latent_dim,) , name = 'Decoder_Model_input')
D = Dense(Initial_Shape[1] * Initial_Shape[2] * Initial_Shape[3])(D_Input)
D = Reshape(target_shape = ( Initial_Shape[1], Initial_Shape[2], Initial_Shape[3]))(D)
D = Conv2DTranspose(32,3, padding = 'same', activation = 'relu', strides = (2,2))(D)
D = Conv2DTranspose(Number_of_channels,3,padding = 'same', activation = 'sigmoid')(D)

decoder = Model(D_Input, D)
Reconstructed_Image = decoder(Mapped_Space)

###Class for VAE Loss function
class VAE_loss(keras.layers.Layer):
  def vae_loss(self, E_Input, Reconstructed_Image):
    E_Input = K.flatten(E_Input)
    Reconstruction_Image = K.flatten(Reconstructed_Image)
    # Reconstruction Error 
    R_loss = binary_crossentropy(E_Input, Reconstructed_Image)
    #KL_Divergence
    KL_loss = -0.005 * K.mean(1 + variance - K.square(mean)- K.exp(variance), axis = -1)
    # Total_loss
    Total_loss = K.mean(R_loss + KL_loss)
    return Total_loss

  #keras functioon requires you to call define a function called call
  def call(self,inputs):
    E_Input = inputs[0]
    Reconstructed_Image = inputs[1]
    loss = self.vae_loss(E_Input, Reconstructed_Image)
    self.add_loss(loss, inputs = inputs)
    return E_Input

y = VAE_loss()([E_Input, Reconstructed_Image])  #loss function
vae = Model(E_Input, y)   #define the kera model
vae.compile(optimizer = 'adam', loss = None) #choos the optimizer 
History = vae.fit(x_train, None, epochs= 200, batch_size= 64, validation_split = 0.2)
