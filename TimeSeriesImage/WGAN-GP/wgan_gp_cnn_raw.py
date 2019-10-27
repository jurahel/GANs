
# Large amount of credit goes to:
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# which I've used as a reference for this implementation

from __future__ import print_function, division


from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop,Adam
from keras.constraints import max_norm
from functools import partial

import keras.backend as K
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd


class Crypto():
    def __init__(self,data_path):
        df=pd.read_csv(data_path,sep=';')
        self.data = df[df.Symbol=='BTCUSD'].Close.reset_index(drop=True).head(5000)
        self.ret=self.data.pct_change().dropna()
        
    def scale(self,a,b):
        print('Scale Data ...')
        self.max_ts = self.data.max()
        self.min_ts = self.data.min()
        scaled_data=(b-a)*(self.data-self.min_ts)/(self.max_ts-self.min_ts)+a
        self.scaled_data=scaled_data

    def sequentialize(self,seq_len):
        print('Sequentialize Data ...')
        self.seq_len = seq_len
        seqs=[]
        for i in range(seq_len,self.scaled_data.shape[0]):
            seqs.append(self.scaled_data.iloc[i-seq_len:i].values.tolist())

        self.data_seq = np.array(seqs)

    def reshape_as_img(self):
        imgs =[]
        self.col = int(np.sqrt(self.data_seq.shape[1]))
        self.seq_len = self.col
        print('Reshaping...')
        i=0
        for seq in self.data_seq:

            imgs.append(seq.reshape((self.col,self.seq_len)))

        self.imgs=np.array(imgs)
        print('Done Reshaping')

    def plot_ts(self):
        plt.plot(self.data)
        plt.savefig('images_raw/TimeSeries.png')
        plt.close()

        cols = int(np.floor(np.sqrt(self.data.shape[0])))
        rows = cols

        plt.imshow(self.data[:rows*cols].values.reshape((rows,cols)))
        plt.savefig('images_raw/TimeSeriesImage.png')
        plt.close()


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGANGP():
    def __init__(self,seq_len):
        self.seq_len = seq_len
        self.img_rows = int(np.sqrt(self.seq_len))
        self.img_cols = int(np.sqrt(self.seq_len))
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 15
        optimizer = Adam()

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #                for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)
            
        
        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()
        #model.add(BatchNormalization(momentum =.8,axis=1))
        model.add(Dense(1 * int(self.img_rows/4) * int(self.img_rows/4),kernel_constraint=max_norm(1.), activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((int(self.img_rows/4), int(self.img_rows/4), 1)))
        model.add(UpSampling2D())
        #model.add(BatchNormalization(momentum =.8))
        
        model.add(Conv2D(self.seq_len*2, kernel_size=3, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(UpSampling2D())
        #model.add(BatchNormalization(momentum =.8))
        model.add(Conv2D(self.seq_len*2, kernel_size=3, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        #model.add(BatchNormalization(momentum =.8))
        

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()
        
        model.add(Conv2D(self.seq_len*2, kernel_size=3, strides=1, input_shape=self.img_shape, padding="same"))
        model.add(BatchNormalization(momentum =.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(self.seq_len*2, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum =.8))
        
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(self.seq_len*2, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum =.8))
        
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(self.seq_len*2, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum =.8))
        
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1,kernel_constraint=max_norm(1.)))
        #model.add(Activation('sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, data, epochs, batch_size, sample_interval):
        # Load the dataset
        self.d_loss_series = []
        self.g_loss_series = []
        
        X_train = data.imgs


        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
              
              
        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                                [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)

            # Plot the progress
            self.d_loss_series.append(d_loss[0])
            self.g_loss_series.append(g_loss)


            print ("%d [Critic loss: %f] [Generator loss: %f]" % (epoch, d_loss[0], g_loss))
            
            # If at save interval => save generated image samples
            if (epoch>0)&(epoch % sample_interval == 0):
                self.sample_images(epoch)


    def sample_images(self, epoch):
        r, c = 3,3
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c, figsize=(12,12))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].plot(gen_imgs[cnt, :,:,0].flatten())
                axs[i,j].set_ylim((-1,1))
                
                cnt += 1
        fig.savefig("images_raw/plot_gan_%d.png" % epoch)
        plt.close()
        

        fig, axs = plt.subplots(2, figsize=(12,12))
        
        axs[0].plot(self.g_loss_series)
        axs[0].set_title('Critic Loss')
        axs[1].plot(self.d_loss_series)
        axs[1].set_title('Discriminator Loss')
        fig.savefig("images_raw/losses.png")
        plt.close()
        

if __name__ == '__main__':
    btc= Crypto('data/ETH_BTC.csv')
    #scale data
    btc.scale(-1,1)
    btc.plot_ts()
    #sequentialize the data
    seq_len=64
    btc.sequentialize(seq_len)
    btc.reshape_as_img()

    print(btc.imgs.shape)

    wgan = WGANGP(seq_len=seq_len)
    wgan.train(btc,epochs=30000, batch_size=32, sample_interval=1)