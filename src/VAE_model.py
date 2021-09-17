import tensorflow as tf
keras=tf.keras
import tensorflow.keras.backend as K



class Sample(tf.keras.layers.Layer):
  def call(self,inputs):
    mu,sigma=inputs
    shapes=tf.shape(mu)
    batch=shapes[0]
    col=shapes[1]
    eps=K.random_normal((batch,col))
    Z=mu+eps*tf.exp(sigma*0.5)
    return Z

def encoder_layers(inputs,latent_dim):

  ## Encoder Layers

  x=keras.layers.Dense(int(16),activation='relu')(inputs)
  x=keras.layers.BatchNormalization()(x)
  x=keras.layers.Dense(int(8),activation='relu')(x)
  x=keras.layers.BatchNormalization()(x)
  mu=keras.layers.Dense(latent_dim)(x)
  sigma=keras.layers.Dense(latent_dim)(x)

  return mu,sigma

def encoder_model(input_shape,latent_dim):

  inputs=keras.layers.Input(shape=input_shape)
  mu,sigma=encoder_layers(inputs,latent_dim)

  Z=Sample()([mu,sigma])

  enc_model=keras.models.Model(inputs=inputs,outputs=[mu,sigma,Z])

  return enc_model


def decoder_layers(inputs,cols):

  ## Decoder Layers
  x=keras.layers.Dense(6,activation='relu')(inputs)
  x=keras.layers.BatchNormalization()(x)
  x=keras.layers.Dense(cols,activation='relu')(x)
  x=keras.layers.BatchNormalization()(x)
  x=keras.layers.Dense(cols,activation='linear')(x)

  return x


def decoder_model(latent_dim,cols):
  inputs=keras.layers.Input(shape=latent_dim)
  outpus=decoder_layers(inputs,cols)

  dec_model=keras.models.Model(inputs,outpus)
  return dec_model

def KL_loss(inputs, outputs, mu, sigma):

  los=tf.keras.losses.mean_squared_error(inputs,outputs)
  kl_loss = -0.5 * (1 + sigma - tf.square(mu) - tf.exp(sigma))
  kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
  # kl_loss = (1 + sigma - tf.square(mu) - tf.math.exp(sigma))* -0.5
  return kl_loss+los

def mse_loss(inputs,outputs):

  los=tf.keras.losses.mean_squared_error(inputs,outputs)
  return los

def vae_model(input_shape,encoder,decoder):
  inputs=keras.layers.Input(shape=input_shape)
  mu,sigma,Z=encoder(inputs)
  dec_output=decoder(Z)
  model=keras.models.Model(inputs,dec_output)
  losskl=KL_loss(inputs,dec_output,mu,sigma)
  # mse = tf.keras.losses.mean_squared_error(inputs,dec_output)#mse_loss(inputs,dec_output)
  model.add_loss(losskl)
  # model.add_loss(mse)
  return model
