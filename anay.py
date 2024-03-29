import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

np.random.seed(42)

class Scaler():
	# hint: https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
	def __init__(self):
		raise NotImplementedError
	def __call__(self,features, is_train=False):
		raise NotImplementedError


def get_features(csv_path,is_train=False,scaler=None):
	'''
	Description:
	read input feature columns from csv file
	manipulate feature columns, create basis functions, do feature scaling etc.
	return a feature matrix (numpy array) of shape m x n 
	m is number of examples, n is number of features
	return value: numpy array
	'''

	'''
	Arguments:
	csv_path: path to csv file
	is_train: True if using training data (optional)
	scaler: a class object for doing feature scaling (optional)
	'''

	'''
	help:
	useful links: 
		* https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
		* https://www.geeksforgeeks.org/python-read-csv-using-pandas-read_csv/
	'''
	X = pd.read_csv(csv_path,index_col=0)
	X = X.drop("frp",axis=1)
	X = X.drop("version",axis=1)
	X = X.drop("instrument",axis=1)
	X = X.drop("acq_date",axis=1)
	temp = pd.get_dummies(X["daynight"])
	X = X.drop("daynight",axis=1)
	X["day"] = temp.iloc[:,0]
	X["night"] = temp.iloc[:,1]
	temp1 = pd.get_dummies(X["satellite"])
	X = X.drop("satellite",axis=1)
	X["aqua"] = temp1.iloc[:,0]
	X["terra"] = temp1.iloc[:,1]
	#cprint(X.shape)
	columns = (X.columns[0])
	max = [X[c].max() for c in X.columns]
	min = [X[c].min() for c in X.columns]
	i=0
	for c in X.columns:
		while(i<len(X.columns)):
			X[c] = (X[c]-min[i])/(max[i]-min[i])
			i = i+1
			break
	one = np.ones((X.shape[0],1))
	X = np.hstack((one,X))
	#print(X)
	print(X.shape)
	
	return X
	
	
	

	

   # raise NotImplementedError

def get_targets(csv_path):
	'''
	Description:
	read target outputs from the csv file
	return a numpy array of shape m x 1
	m is number of examples
	'''
	
	targets = pd.read_csv(csv_path)
	targets = targets.iloc[:,-2]
	targets = (targets -targets.min())/(targets.max()-targets.min())
	#print(targets)
	return targets
	#raise NotImplementedError
	 

def analytical_solution(feature_matrix, targets, C=0.0):
	'''
	Description:
	implement analytical solution to obtain weights
	as described in lecture 5d
	return value: numpy array
	'''

	'''
	Arguments:
	feature_matrix: numpy array of shape m x n
	targets: numpy array of shape m x 1
	'''
	
	A = np.linalg.inv(np.dot(feature_matrix.T,feature_matrix))
	B = np.dot(feature_matrix.T,targets)
	W = np.dot(A,B)
	#print(W)
	#print(W.shape)
	return W
   # raise NotImplementedError 

def get_predictions(feature_matrix, weights):
	'''
	description
	return predictions given feature matrix and weights
	return value: numpy array
	'''

	'''
	Arguments:
	feature_matrix: numpy array of shape m x n
	weights: numpy array of shape n x 1
	'''
	
	print(feature_matrix.shape,weights.shape)
	# print(feature_matrix)
	Ypred = np.dot(feature_matrix,weights)
	
	#print(Ypred)
	return Ypred
   # raise NotImplementedError

def mse_loss(feature_matrix, weights, targets):
	'''
	Description:
	Implement mean squared error loss function
	return value: float (scalar)
	'''

	'''
	Arguments:
	feature_matrix: numpy array of shape m x n
	weights: numpy array of shape n x 1
	targets: numpy array of shape m x 1
	'''
	
	Ypred  = np.dot(feature_matrix,weights)
	
	
	loss = (np.sum(Ypred - targets)**2)/(feature_matrix.shape[0])
	
	return loss
	#raise NotImplementedError

def l2_regularizer(weights):
	'''
	Description:
	Implement l2 regularizer
	return value: float (scalar)
	'''

	'''
	Arguments
	weights: numpy array of shape n x 1
	'''
	reg = np.sum(weights**2)
	return reg
	#print(reg)
	#raise NotImplementedError

def loss_fn(feature_matrix, weights, targets, C=0.0):
	'''
	Description:
	compute the loss function: mse_loss + C * l2_regularizer
	'''

	'''
	Arguments:
	feature_matrix: numpy array of shape m x n
	weights: numpy array of shape n x 1
	targets: numpy array of shape m x 1
	C: weight for regularization penalty
	return value: float (scalar)
	'''
	mse = mse_loss(feature_matrix,weights,targets) 
	l2 = l2_regularizer(weights)
	loss = mse + C*l2
	print("------loss func",loss)
	return loss
	
	#raise NotImplementedError

def compute_gradients(feature_matrix, weights, targets, C=0.0):
	'''
	Description:
	compute gradient of weights w.r.t. the loss_fn function implemented above
	'''

	'''
	Arguments:
	feature_matrix: numpy array of shape m x n
	weights: numpy array of shape n x 1
	targets: numpy array of shape m x 1
	C: weight for regularization penalty
	return value: numpy array
	'''
	# gradients = np.dot(feature_matrix,loss_fn(feature_matrix,weights,targets,C))/(2*feature_matrix.shape[0])
	predictions = np.dot(feature_matrix,weights)
	gradients = (np.dot(feature_matrix.T,(predictions-targets)))+(C*l2_regularizer(weights))/(2*feature_matrix.shape[0])
	return gradients

def sample_random_batch(feature_matrix, targets, batch_size):
	'''
	Description
	Batching -- Randomly sample batch_size number of elements from feature_matrix and targets
	return a tuple: (sampled_feature_matrix, sampled_targets)
	sampled_feature_matrix: numpy array of shape batch_size x n
	sampled_targets: numpy array of shape batch_size x 1
	'''

	'''
	Arguments:
	feature_matrix: numpy array of shape m x n
	targets: numpy array of shape m x 1
	batch_size: int
	'''    
	featureData=[]
	targetData=[]
	mydata=[]
	for x in range(feature_matrix.shape(0)): 
		mydata.append(tuple(targets[x],feature_matrix[x]))
	picks =	np.random.randint(feature_matrix.shape[0],size = batch_size)
	for x in picks:
		featureData.append(mydata[x][0])
		targetData.append(mydata[x][1])
	print(picks,targetData)
	return featureData,targetData
	
def initialize_weights(n):
	'''
	Description:
	initialize weights to some initial values
	return value: numpy array of shape n x 1
	'''

	'''
	Arguments
	n: int
	'''
	return np.random.rand(n,1)

def update_weights(weights, gradients, lr):
	'''
	Description:
	update weights using gradient descent
	retuen value: numpy matrix of shape nx1
	'''

	'''
	Arguments:
	# weights: numpy matrix of shape nx1
	# gradients: numpy matrix of shape nx1
	# lr: learning rate
	'''    
	weights = weights - lr*gradients 
	return weights
 
	#raise NotImplementedError

def early_stopping(arg_1=None, arg_2=None, arg_3=None, arg_n=None):
	# allowed to modify argument list as per your need
	# return True or False
	raise NotImplementedError
	

def plot_trainsize_losses():
	'''
	Description:
	plot losses on the development set instances as a function of training set size 
	'''

	'''
	Arguments:
	# you are allowed to change the argument list any way you like 
	'''    

	raise NotImplementedError


def do_gradient_descent(train_feature_matrix,  
						train_targets, 
						dev_feature_matrix,
						dev_targets,
						lr=1.0,
						C=0.0,
						batch_size=32,
						max_steps=10000,
						eval_steps=5):
	'''
	feel free to significantly modify the body of this function as per your needs.
	** However **, you ought to make use of compute_gradients and update_weights function defined above
	return your best possible estimate of LR weights

	a sample code is as follows -- 
	'''
	n=train_feature_matrix.shape[1]
	weights = initialize_weights(13)
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

		'''
		implement early stopping etc. to improve performance.
		'''

	return weights

def do_evaluation(feature_matrix, targets, weights):
	# your predictions will be evaluated based on mean squared error 
	predictions = get_predictions(feature_matrix, weights)
	loss =  mse_loss(feature_matrix, weights, targets)
	return loss

if __name__ == '__main__':
	#scaler = Scaler() #use of scaler is optional
	scaler = False
	train_features, train_targets = get_features('./data/train.csv',True,scaler), get_targets('./data/train.csv')
	dev_features, dev_targets = get_features('./data/dev.csv',False,scaler), get_targets('./data/dev.csv')

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
						lr=1.0,
						C=0.0,
						batch_size=32,
						max_steps=2000000,
						eval_steps=5)

	print('evaluating iterative_solution...')
	dev_loss=do_evaluation(dev_features, dev_targets, gradient_descent_soln)
	train_loss=do_evaluation(train_features, train_targets, gradient_descent_soln)
	print('gradient_descent_soln \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))
	#plot_trainsize_losses()  