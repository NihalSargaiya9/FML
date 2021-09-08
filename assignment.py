# -*- coding: utf-8 -*-
"""assignment.ipynb

Automatically generated by Colaboratory.

Original file is located at
	https://colab.research.google.com/drive/18EpAiH6VNOOkIPXZsW5wXOCbo4X4QzCd
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

np.random.seed(42)
mydata = []
class Scaler():
	# hint: https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
	# def __init__(self):
	#     raise NotImplementedError
	def __call__(self,features, is_train=False):
		'''
		Arguments : features N X D all are numeric
		'''
		#Appying min max normalization --> May be if outliers are present might effect accuracy
		#Formula (x-min)/(max-min)
		#Gets min across all columns
		min=features.min(0)
		#Gets max-min across all columns
		ptp=features.ptp(0)
		features_normed=(features-min)/ptp
		return features_normed

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
	feaures = ["scan","track","bright_t31","daynight","latitude","longitude","brightness","confidence","satellite"]
	data = pd.read_csv(csv_path)[feaures]
	X = data
	temp = pd.get_dummies(X["daynight"])
	X = X.drop("daynight",axis=1)
	X["day"]=temp.iloc[:,0]
	X["night"]=temp.iloc[:,1]
	temp = pd.get_dummies(X["satellite"])
	X = X.drop("satellite",axis=1)
	X["aqua"]=temp.iloc[:,0]
	X["terra"]=temp.iloc[:,1]
	# scaler(X) 
	# for _ in range(X.shape[1]):

	# 	# scalingData[feaures[_]]["u"]=u
	# 	# scalingData[feaures[_]]["std"]=std
	# 	X.iloc[:,_]=(X.iloc[:,_]-u)/std
	u = np.mean(X,0)
	std = np.std(X,0)
	X = (X-u)/std
	print(X)
	one = np.ones((X.shape[0],1))
	X = np.hstack((one,X))
	# X = squareBasis(X)
	return X

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
	A = np.linalg.inv(np.dot(feature_matrix.T,feature_matrix)+C*l2_regularizer(targets))
	B = np.dot(feature_matrix.T,targets)

	W = np.dot(A,B)

	return W
	'''
	Arguments:
	feature_matrix: numpy array of shape m x n
	targets: numpy array of shape m x 1
	'''

def get_predictions(feature_matrix, weights):
	'''
	description
	return predictions given feature matrix and weights
	return value: numpy array
	'''
	Ypred=np.dot(feature_matrix,weights)
	return abs(Ypred)
	'''
	Arguments:
	feature_matrix: numpy array of shape m x n
	weights: numpy array of shape n x 1
	'''

def mse_loss(feature_matrix, weights, targets):
	'''
	Description:
	Implement mean squared error loss function
	return value: float (scalar)
	'''
	Ypred  = get_predictions(feature_matrix,weights)
	targets = targets.values
	targets.reshape(targets.shape[0],1)
	loss = np.sum((Ypred - targets)**2)/(2*(feature_matrix.shape[0]))
	return loss
	'''
	Arguments:
	feature_matrix: numpy array of shape m x n
	weights: numpy array of shape n x 1
	targets: numpy array of shape m x 1
	'''

def l2_regularizer(weights):
	'''
	Description:
	Implement l2 regularizer
	return value: float (scalar)
	'''

	reg = np.sum(weights**2)**(0.5)
	return reg
	'''
	Arguments
	weights: numpy array of shape n x 1
	'''

def loss_fn(feature_matrix, weights, targets, C=0.0):
	'''
	Description:
	compute the loss function: mse_loss + C * l2_regularizer
	'''
	
	mse = mse_loss(feature_matrix,weights,targets) 
	l2 = l2_regularizer(weights)
	loss = mse + C*l2
	# print("------loss func",loss)
	return loss
	'''
	Arguments:
	feature_matrix: numpy array of shape m x n
	weights: numpy array of shape n x 1
	targets: numpy array of shape m x 1
	C: weight for regularization penalty
	return value: float (scalar)
	'''

def compute_gradients(feature_matrix, weights, targets, C=0.0):
	'''
	Description:
	compute gradient of weights w.r.t. the loss_fn function implemented above
	'''
	predictions = get_predictions(feature_matrix,weights)
	predictions = predictions.reshape(predictions.shape[0],1)
	targets = targets.reshape(predictions.shape[0],1)

	# print(feature_matrix.shape,targets.shape,weights.shape,predictions.shape,np.subtract(predictions,targets).shape)
	
	temp = (np.dot(feature_matrix.T,np.subtract(predictions,targets)))

	reg = (C*l2_regularizer(weights))
	# reg = (C*np.sum(weights**2))
	
	# print(temp.shape,reg.shape)
	gradients = (temp+reg)/(2*feature_matrix.shape[0])
	
	return gradients

	'''
	Arguments:
	feature_matrix: numpy array of shape m x n
	weights: numpy array of shape n x 1
	targets: numpy array of shape m x 1
	C: weight for regularization penalty
	return value: numpy array
	'''

import random
import numpy as np
def sample_random_batch(feature_matrix, targets, batch_size):
	'''
	Description
	Batching -- Randomly sample batch_size number of elements from feature_matrix and targets
	return a tuple: (sampled_feature_matrix, sampled_targets)
	sampled_feature_matrix: numpy array of shape batch_size x n
	sampled_targets: numpy array of shape batch_size x 1
	'''
	#Getting random batch_size number indexes from 0 to n
	featureData=[]
	targetData=[]
	mydata =[]
	if(len(mydata)==0):
		for x in range(feature_matrix.shape[0]): 
			mydata.append((targets[x],feature_matrix[x]))

	# print("size of feature matrix :",feature_matrix.shape)

	picks =	np.random.randint(feature_matrix.shape[0],size = batch_size)
	# print(picks,max(picks),batch_size)
	for x in picks:
		featureData.append(list(mydata[x][1]))
		targetData.append(mydata[x][0])
 
	return np.array(featureData),np.array(targetData)
	'''
	Arguments:
	feature_matrix: numpy array of shape m x n
	targets: numpy array of shape m x 1
	batch_size: int
	'''    
	#raise NotImplementedError
#print(sample_random_batch(np.arange(100).reshape(10,10),np.arange(10),2))

def initialize_weights(n):
	'''
	Description:
	initialize weights to some initial values
	return value: numpy array of shape n x 1
	'''
	w = np.zeros(n)
	return w
	'''
	Arguments
	n: int
	'''

def update_weights(weights, gradients, lr):
	'''
	Description:
	update weights using gradient descent
	retuen value: numpy matrix of shape nx1
	'''
	N=weights.shape[0]

	#Updating weights by given formula
	for i in range(0,N):
		weights[i] = weights[i] - lr*(1/N)*gradients[i]

	return weights
	'''
	Arguments:
	# weights: numpy matrix of shape nx1
	# gradients: numpy matrix of shape nx1
	# lr: learning rate
	'''    
# import numpy as np
# print(update_weights(np.arange(10),np.random.rand(10),0.0001))

def early_stopping(patience=None):
	# allowed to modify argument list as per your need
	# return True or False
	if patience >= 800:
	   return True
	else:
	   return False

def plot_trainsize_losses(x_data, y_data, title):
	'''
	Description:
	plot losses on the development set instances as a function of training set size 
	'''
	plt.plot(x_data, y_data, 'b')
	plt.title(title)
	plt.xlabel('Training Samples Count')
	plt.ylabel('Dev Loss')
	plt.legend()
	plt.show()
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
	check = 0
	loss = np.inf
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
		'''
		implement early stopping etc. to improve performance.
		'''
		if loss <= dev_loss:
			  check += 1
		else:
				loss = dev_loss
				best_weights = weights
				check= 0
		if early_stopping(check):
				print(f'Stopping at {step} steps')
				break

	return best_weights

def do_evaluation(feature_matrix, targets, weights):
	# your predictions will be evaluated based on mean squared error 
	predictions = get_predictions(feature_matrix, weights)
	loss =  mse_loss(feature_matrix, weights, targets)
	return loss
if __name__ == '__main__':
	# scaler = Scaler() #use of scaler is optional
	scaler = False
	train_features, train_targets = get_features('./train.csv',True,scaler), get_targets('./train.csv')
	dev_features, dev_targets = get_features('./dev.csv',False,scaler), get_targets('./dev.csv')
	test_features = get_features('./test.csv',False,scaler)
	# trainYTargets = pd.read_csv("./data/train.csv")["frp"]
	# devYTargets = pd.read_csv("./data/dev.csv")["frp"]


	a_solution = analytical_solution(train_features, train_targets, C=1e-8)
	print('evaluating analytical_solution...')
	
	dev_loss=do_evaluation(dev_features, dev_targets, a_solution)
	train_loss=do_evaluation(train_features, train_targets, a_solution)
	print('analytical_solution \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))
	print('training LR using gradient descent...')
		
	gradient_descent_soln = finalSol

	predictions = get_predictions(test_features, a_solution)
	id = np.arange(test_features.shape[0])
	df = pd.DataFrame({"frp":predictions})
	df.rename(columns={"":"id"})
	df.to_csv("answer.csv")
	# gradient_descent_soln = 0
	# exit(0)
	gradient_descent_soln = do_gradient_descent(train_features, 
						train_targets, 
						dev_features,
						dev_targets,
						lr=0.05,
						C=1e-8,
						batch_size=32,
						max_steps=200000,
						eval_steps=5)

	# print('evaluating iterative_solution...')
	# dev_loss=do_evaluation(dev_features, devYTargets, finalSol)
	# train_loss=do_evaluation(train_features, trainYTargets, finalSol)

	# predictions = get_predictions(test_features, finalSol)
	# id = np.arange(test_features.shape[0])
	# df = pd.DataFrame({"frp":predictions})
	# df.rename(columns={"":"id"})
	# df.to_csv("answer.csv")

	print('gradient_descent_soln \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))
	plot_trainsize_losses()  