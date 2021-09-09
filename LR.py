import pandas as pd
import numpy as np
import matplotlib
import random
from matplotlib import pyplot as plt
matplotlib.use("Agg")


np.random.seed(42)
class Scaler():
	def __call__(self,features, is_train=False):
		'''
		Arguments : features N X D all are numeric
		'''
		min=features.min(0)
		ptp=features.ptp(0)
		features_normed=(features-min)/ptp
		return features_normed

# Polynomial Basis function

def polyBasis(X):
	for x in range(X.shape[1]):
		X[:,x] = X[:,x]**x
	return X

# Square Basis function

def squareBasis(X):
	np.power(X,2)
	return X



def get_features(csv_path,is_train=False,scaler=None):
	feaures = ["scan","track","bright_t31","daynight","latitude","longitude","brightness","confidence","satellite"]
	data = pd.read_csv(csv_path)[feaures]
	X = data

	# Onehot encoding for Daynight
	temp = pd.get_dummies(X["daynight"])
	X = X.drop("daynight",axis=1)
	X["day"]=temp.iloc[:,0]
	X["night"]=temp.iloc[:,1]
	# Onehot encoding for satellite
	temp = pd.get_dummies(X["satellite"])
	X = X.drop("satellite",axis=1)
	X["aqua"]=temp.iloc[:,0]
	X["terra"]=temp.iloc[:,1]
	u = np.mean(X,0)
	std = np.std(X,0)
	X = (X-u)/std

	one = np.ones((X.shape[0],1))
	X = np.hstack((one,X))

	# X = polyBasis(X)			#uncomment to apply basis function 1
	# X = squareBasis(X)		#uncomment to apply basis function 2

	return X

def get_targets(csv_path):
	'''
	Description:
	read target outputs from the csv file
	return a numpy array of shape m x 1
	m is number of examples
	'''
	df = pd.read_csv(csv_path)
	return df.frp

def analytical_solution(feature_matrix, targets, C=0.0):
	A = np.linalg.inv(np.dot(feature_matrix.T,feature_matrix)+C*l2_regularizer(targets))
	B = np.dot(feature_matrix.T,targets)
	W = np.dot(A,B)
	return W


def get_predictions(feature_matrix, weights):
	Ypred=np.dot(feature_matrix,weights)
	return abs(Ypred)


def mse_loss(feature_matrix, weights, targets):
	Ypred  = get_predictions(feature_matrix,weights)
	targets = targets.values
	targets.reshape(targets.shape[0],1)
	loss = np.sum((Ypred - targets)**2)/(2*(feature_matrix.shape[0]))
	return loss

def l2_regularizer(weights):
	reg = np.sum(weights**2)**(0.5)
	return reg

def loss_fn(feature_matrix, weights, targets, C=0.0):
	mse = mse_loss(feature_matrix,weights,targets) 
	l2 = l2_regularizer(weights)
	loss = mse + C*l2
	return loss

def compute_gradients(feature_matrix, weights, targets, C=0.0):
	predictions = get_predictions(feature_matrix,weights)
	predictions = predictions.reshape(predictions.shape[0],1)
	targets = targets.reshape(predictions.shape[0],1)
	temp = (np.dot(feature_matrix.T,np.subtract(predictions,targets)))
	reg = (C*l2_regularizer(weights))
	gradients = (temp+reg)/(2*feature_matrix.shape[0])
	return gradients

def sample_random_batch(feature_matrix, targets, batch_size):
	sampled_targets=[]
	indexes=random.sample(range(0, feature_matrix.shape[0]), batch_size)
	new_feature_matrix=[]
	for i in indexes:
	  new_feature_matrix.append(feature_matrix[i])
	  count = 0
	  sampled_targets.append(targets[i])
	return np.array(new_feature_matrix),np.array(sampled_targets)

def initialize_weights(n):
	w = np.zeros(n)
	return w

def update_weights(weights, gradients, lr):
	N=weights.shape[0]

	for i in range(0,N):
		weights[i] = weights[i] - lr*(1/N)*gradients[i]

	return weights

def early_stopping(patience=0):
	if patience >700:
	   return True
	return False

def plot_trainsize_losses(x_data, y_data,t):
	# %matplotlib inline
	print(x_data,y_data)
	plt.plot(x_data, y_data)
	plt.xlabel('Training Samples Count')
	plt.ylabel('Dev Loss')
	plt.title(t)
	plt.savefig('plot1.png')
	plt.show()
	return 

def do_gradient_descent(train_feature_matrix,  
						train_targets, 
						dev_feature_matrix,
						dev_targets,
						lr=1.0,
						C=0.0,
						batch_size=32,
						max_steps=1000,
						eval_steps=5):
	check = 0
	loss =np.inf
	best_weights= None
	n = len(train_feature_matrix[0])
	weights = initialize_weights(n)
	dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
	train_loss = mse_loss(train_feature_matrix, weights, train_targets)
	print("step {} \t dev loss: {} \t train loss: {}".format(0,dev_loss,train_loss))
	for step in range(1,max_steps+1):

		#sample a batch of features and gradients
		features,targets = sample_random_batch(train_feature_matrix,train_targets,batch_size)
		
		#compute gradients
		gradients = compute_gradients(features, weights, targets, C)
		
		#update weights
		weights = update_weights(weights, gradients, lr)

		if step%eval_steps == 0:
			dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
			train_loss = mse_loss(train_feature_matrix, weights, train_targets)
			print("step {} \t dev loss: {} \t train loss: {}".format(step,dev_loss,train_loss))
		if early_stopping(check):
				print(f'Stopping at {step} steps')
				break
		if loss <= dev_loss:
			  check += 1
		else:
				loss = dev_loss
				best_weights = weights
				check= 0
	return best_weights

def do_evaluation(feature_matrix, targets, weights):
	predictions = get_predictions(feature_matrix, weights)
	loss =  mse_loss(feature_matrix, weights, targets)
	return loss
if __name__ == '__main__':
	scaler = False
	train_features, train_targets = get_features('./data/train.csv',True,scaler), get_targets('./data/train.csv')
	dev_features, dev_targets = get_features('./data/dev.csv',False,scaler), get_targets('./data/dev.csv')
	test_features = get_features('./data/test.csv',False,scaler)
	a_solution = analytical_solution(train_features, train_targets, C=1e-8)
	print('evaluating analytical_solution...')
	dev_loss=do_evaluation(dev_features, dev_targets, a_solution)
	train_loss=do_evaluation(train_features, train_targets, a_solution)
	print('analytical_solution \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))
	print('training LR using gradient descent...')


	gradient_descent_soln = do_gradient_descent(train_features, 
						train_targets, 
						dev_features,
						dev_targets,
						lr=0.0009,
						C=1e-8,
						batch_size=50,
						max_steps=400000,
						eval_steps=20)
						
	print('evaluating iterative_solution...')
	dev_loss=do_evaluation(dev_features, dev_targets, gradient_descent_soln)
	train_loss=do_evaluation(train_features, train_targets, gradient_descent_soln)

	print('gradient_descent_soln \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))