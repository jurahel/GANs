from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LSTM
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam

import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd


class Crypto():
	def __init__(self,name,data):
		df = pd.read_csv('ETH_BTC.csv',sep=';')
		self.name = name
		self.data = data
		self.ret=(self.data.pct_change().dropna()+1).apply(np.log)

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
		

class WGAN():
	def __init__(self,seq_len):
		self.seq_len = seq_len
		self.no_seqs = 1
		self.channels = 1
		self.img_shape = (self.seq_len, self.no_seqs)#,self.channels)
		self.latent_dim = self.seq_len

		# Following parameter and optimizer set as recommended in paper
		self.n_critic = 5
		self.clip_value = 0.01
		optimizer = Adam()# standard: lr=0.00005

		# Build and compile the critic
		self.critic = self.build_critic()
		self.critic.compile(loss=self.wasserstein_loss,
			optimizer=optimizer,
			metrics=['accuracy'])

		# Build the generator
		self.generator = self.build_generator()

		# The generator takes noise as input and generated imgs
		z = Input(shape=(self.latent_dim,1))
		img = self.generator(z)
		
		# For the combined model we will only train the generator
		self.critic.trainable = False
		
		# The critic takes generated images as input and determines validity
		valid = self.critic(img)

		# The combined model  (stacked generator and critic)
		self.combined = Model(z, valid)
		self.combined.compile(loss=self.wasserstein_loss,
			optimizer=optimizer,
			metrics=['accuracy'])

	def wasserstein_loss(self, y_true, y_pred):
		return K.mean(y_true * y_pred)

	def build_generator(self):

		model = Sequential()
		model.add(LSTM(500,return_sequences=True,input_shape=(self.latent_dim,1)))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Dropout(0.15))
		model.add(LSTM(500))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Dense(btc.seq_len))
		model.add(Activation('sigmoid'))
		model.add(Reshape((btc.seq_len,1)))

		noise = Input(shape=(self.latent_dim,1))
		img = model(noise)	
		#model.summary()

		return Model(noise, img)

	def build_critic(self):

		model = Sequential()
		model.add(LSTM(500,return_sequences=True, input_shape=(btc.data_seq.shape[1],1)))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Dropout(0.15))
		model.add(LSTM(500))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Dense(1))
		model.add(Activation('tanh'))

		img = Input(shape=[btc.data_seq.shape[1],1])
		#model.summary()
		
		validity = model(img)
		return Model(img, validity)

	def train(self, epochs, batch_size, sample_interval,flip_interval):
		self.flip_interval=flip_interval
		self.batch_size=batch_size
		# Load the dataset
		d_loss_series = []
		d_loss_real_series = []
		d_loss_fake_series = []
		g_loss_series = []



		# Rescale -1 to 1
		X_train = btc.data_seq
		

		# Adversarial ground truths
		valid = -np.ones((batch_size, 1))
		fake = np.ones((batch_size, 1))

		ri = np.random.randint(0,self.flip_interval) #select random integer
		for epoch in range(epochs):
			state_ig=epoch
			no=np.random.randint(0,self.flip_interval)

			if no==ri:
				valid=valid*-1
				fake=fake*-1
				state_ig=epoch+1
				print('Flip')

			for _ in range(self.n_critic):

				# ---------------------
				#  Train Discriminator
				# ---------------------

				# Select a random batch of images
				idx = np.random.randint(0, btc.data_seq.shape[0],batch_size)
				imgs = X_train[idx]
				
				
				# Sample noise as generator input
				noise = np.random.normal(0, 1, (batch_size,self.latent_dim,1))

				# Generate a batch of new images
				gen_imgs = self.generator.predict(noise)
				
				imgs = np.expand_dims(imgs,axis=2)
				
				# Train the critic
				d_loss_real = self.critic.train_on_batch(imgs, valid)
				d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
				d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
				
				# Clip critic weights
				for l in self.critic.layers:
					weights = l.get_weights()
					weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
					l.set_weights(weights)


			# ---------------------
			#  Train Generator
			# ---------------------

			g_loss = self.combined.train_on_batch(noise, valid)
			
			if state_ig==epoch:#in case this is true ignore the 
				g_loss_series.append(1-g_loss[0])
				d_loss_series.append(1-d_loss[0])

			flip=False

			g_loss_series.append(1-g_loss[0])
			d_loss_series.append(1-d_loss[0])
			# Plot the progress
			print ("%d [D loss: %f] [G loss: %f]" % (epoch,1-d_loss[0], 1-g_loss[0]))
			
			if (epoch%sample_interval==0)&(epoch>0):
				noise = np.random.normal(0, 1, (batch_size,self.latent_dim,1))
				gen_imgs = self.generator.predict(noise)

				#create a figure with random selection 
				fig, ax = plt.subplots(2)
				fig.suptitle('GAN Simulation - Epoch '+str(epoch))
				dim = int(np.sqrt(btc.seq_len))
				idx=np.random.randint(0,self.batch_size)
				ax[0].imshow(gen_imgs[idx].reshape((dim,dim)))#,vmin=-1,vmax=1)
				ax[1].plot(gen_imgs[idx].flatten())
				fig.savefig('WGAN_Images/WGAN_Pic_'+str(epoch)+'.png')
				plt.close()
				
				g_errors = pd.Series(g_loss_series)
				d_errors = pd.Series(d_loss_series)
				

				fig, ax = plt.subplots(2)
				ax[0].plot(g_errors,color='blue')
				ax[0].legend('G_Error')
				ax[1].plot(d_errors,color='orange')
				ax[1].legend('D_Error')

				plt.savefig('WGAN_Images/g_d_loss.png')
				plt.close()


if __name__ == '__main__':
	#prepare data
	df=pd.read_csv('ETH_BTC.csv',sep=';')
	btc=Crypto('Bitcoin',df[df.Symbol=='BTCUSD'].Close.reset_index(drop=True))
	seq_len = 144
	#btc.data=btc.ret
	btc.scale(-1,1)
	btc.sequentialize(seq_len)
	
	#set up the gan
	wgan = WGAN(btc.seq_len)
	wgan.train(epochs=4000, batch_size=10, sample_interval=10, flip_interval=50)








