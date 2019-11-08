import sys
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import torch.distributions as tdist
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import imageio

np.set_printoptions(precision=4)

class crypto:
	def __init__(self,name,data):
		self.name = name
		self.data = data
		self.skew = self.data.skew()
		self.kurt = self.data.kurtosis()


class DiscriminatorNet(torch.nn.Module):
	"""
	A three hidden-layer discriminative neural network
	"""
	def __init__(self):
		super(DiscriminatorNet, self).__init__()#has the same properties as the torch.nn.module
		n_features = 998
		n_out = 1
		layer1_neurons = 300
		layer2_neurons = 150
		layer3_neurons = 75

		self.hidden0 = nn.Sequential(#Modules will be added to it in the order they are passed in the constructor
		nn.Linear(n_features, layer1_neurons),
		nn.LeakyReLU(0.2),
		nn.Dropout(0.2)
		)
		self.hidden1 = nn.Sequential(
		nn.Linear(layer1_neurons, layer2_neurons),
		nn.LeakyReLU(0.2),
		nn.Dropout(0.2)
		)
		self.hidden2 = nn.Sequential(
		nn.Linear(layer2_neurons, layer3_neurons),
		nn.LeakyReLU(0.2),
		nn.Dropout(0.2)
		)
		self.out = nn.Sequential(
		torch.nn.Linear(layer3_neurons, n_out),
		torch.nn.Sigmoid()
		)
		self.requires_grad=True

	def forward(self, x):
		x = self.hidden0(x)
		x = self.hidden1(x)
		x = self.hidden2(x)
		x = self.out(x)
		return x

class GeneratorNet(torch.nn.Module):
	"""
	A three hidden-layer generative neural network
	"""
	def __init__(self):
		super(GeneratorNet, self).__init__()
		n_features = 10
		n_out = 998
		layer1_neurons = 100
		layer2_neurons = 200
		layer3_neurons = 300

		self.hidden0 = nn.Sequential(
		nn.Linear(n_features, layer1_neurons),
		nn.LeakyReLU(0.2)
		)
		self.hidden1 = nn.Sequential(	
		nn.Linear(layer1_neurons, layer2_neurons),
		nn.LeakyReLU(0.2)
		)
		self.hidden2 = nn.Sequential(
		nn.Linear(layer2_neurons, layer3_neurons),
		nn.LeakyReLU(0.2)
		)
		
		self.out = nn.Sequential(
		nn.Linear(layer3_neurons, n_out),
		nn.Tanh()
		)
		# is used when net is setup 
	def forward(self, x):
		x = self.hidden0(x)
		x = self.hidden1(x)
		x = self.hidden2(x)
		x = self.out(x)
		return x


# Noise
def noise(size):
	n = Variable(tdist.Normal(0,1).sample((1,size)))
	n_std = 2*(n-n.min())/(n.max()-n.min())-1
	return n
def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size))
    return data
def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size))
    return data
def real_data_target(size):
	'''
	Tensor containing ones, with shape = size
	'''
	data = Variable(torch.ones(size))
	return(data)
def fake_data_target(size):
	'''
	Tensor containing zeros, with shape = size
	'''
	data = Variable(torch.zeros(size))
	return(data)

def pretrain_discriminator(optimizer,real_data):
	optimizer.zero_grad()
	prediction_real = discriminator(real_data)
	error_real = loss(prediction_real, ones_target(1))
	
	error_real.backward()

	#print(error_real.grad)
	
	optimizer.step()
	return(error_real)

def train_discriminator(optimizer, real_data, fake_data):
	
	N = real_data.size(0)

	# Reset gradients
	optimizer.zero_grad()

	# 1.1 Train on Real Data
	prediction_real = discriminator(real_data)

	# Calculate error and backpropagate

	error_real = loss(prediction_real, ones_target(1))
	error_real.backward()
	grad = [p.grad for p in list(discriminator.parameters())]

	# 1.2 Train on Fake Data
	prediction_fake = discriminator(fake_data)
	# Calculate error and backpropagate
	error_fake = loss(prediction_fake, zeros_target(1))
	error_fake.backward()

	# 1.3 Update weights with gradients
	optimizer.step()

	# Return error and predictions for real and fake inputs
	return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer, fake_data):
	# 2. Train Generator
	# Reset gradients
	optimizer.zero_grad()
	# Sample noise and generate fake data
	prediction = discriminator(fake_data)
	# Calculate error and backpropagate
	error = loss(prediction, real_data_target(prediction.size(0)))
	error.backward()
	# Update weights with gradients
	optimizer.step()
	# Return error
	return error

def scale(data_array):
	data_min = data_array.min()
	data_max = data_array.max()
	data_scaled = 2*(data_array-data_min)/(data_max-data_min)-1
	return data_scaled, data_min, data_max

def sample_batch(array_1,array_2,seq_length):
	end_index=np.random.randint(seq_length,array_1.shape[0])
	start_index=end_index-seq_length
	batch = np.array([arr1,arr2]).flatten()
	return batch

'''
df=pd.read_csv('ETH_BTC.csv',sep=';')
eth=crypto('Ether',df[df.Symbol=='ETHUSD'])
btc=crypto('Bitcoin',df[df.Symbol=='BTCUSD'])
#calculate log returns
eth_ret=(eth.data.Close.pct_change().dropna()+1).apply(np.log)
btc_ret=(btc.data.Close.pct_change().dropna()+1).apply(np.log)
#scale data in range -1 to 1
eth_ret_scaled,eth_ret_min,eth_ret_max = scale(eth_ret)
btc_ret_scaled,btc_ret_min,btc_ret_max = scale(btc_ret)
'''
df=np.load('testarray.npy')
seq_length=1000


discriminator = DiscriminatorNet()
generator = GeneratorNet()
# Optimizers
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

# Loss function
loss = nn.BCELoss()

# Number of steps to apply to the discriminator
d_steps = 1  # In Goodfellow et. al 2014 this variable is assigned to 1
# Number of epochs
num_epochs = 20000
pretrain_d = 200


d_errors=[]
g_errors=[]


print('Pretrain Discriminator ... ')
for pte in range(pretrain_d):
	real_data = torch.tensor(df[np.random.randint(0,df.shape[0])],dtype=torch.float)
	
	pretrain_discriminator(d_optimizer,real_data)
	if pte%100==0:
		print(pte)

print('Train Real Model ... ')

for epoch in range(num_epochs):
	if epoch%100 == 0:
		print(epoch)
	
	real_data = torch.tensor(df[np.random.randint(0,df.shape[0])],dtype=torch.float)

	# 1. Train Discriminator	  
	# Generate fake data
	fake_data = generator(noise(10)).detach()#detach means do not compute gradients
	
	# Train D
	d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer,real_data, fake_data)

	# 2. Train Generator
	# Generate fake data
	fake_data = generator(noise(10))
	# Train G
	g_error = train_generator(g_optimizer, fake_data)
	
	#
	d_errors.append(d_error.item())
	g_errors.append(g_error.item())



g_errors = pd.Series(g_errors)
g_errors.rolling(10)
d_errors = pd.Series(d_errors)
d_errors.rolling(10)

fig, ax = plt.subplots(2)
ax[0].plot(g_errors,color='blue')
ax[0].legend('G_Error')
ax[1].plot(d_errors,color='orange')
ax[1].legend('D_Error')

plt.savefig('g_d_loss.png')
plt.show()
plt.close()



res_fake=[]
res_real=[]

samples=1000

for i in range(samples):
	res_fake = np.array((generator(noise(10)).detach().data.tolist())).flatten()
	res_fake = res_fake.reshape((2,-1))
	eth_rets = res_fake[0,:]
	btc_rets = res_fake[1,:]

	bins=np.arange(-1,1.1,0.1)
	freq,btc_bins,eth_bins = np.histogram2d(eth_rets,btc_rets,bins=(bins,bins),normed=True)
	X,Y = np.meshgrid(btc_bins[:-1],eth_bins[:-1])
	Z = freq
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot_surface(X, Y, Z)
	ax.set_zlim((0,60))
	ax.set_title('GAN Sample PDF - ETH and BTC {}/{}'.format(i,samples))
	plt.savefig('Sample_GAN_{}.png'.format(i))
	plt.close()

##Create GIF from simulated data
image_list=['Sample_GAN_'+str(x)+'.png' for x in range(samples)]

images = []
duration=0.2
for image in image_list:
	print(image)
	try:
		images.append(imageio.imread(image))
	except:
		print('missing '+image)


imageio.mimsave("C:/Users/Hellermann/Documents/CryptoCurrencyData/sample_gan_movie.gif",images,duration=duration)
