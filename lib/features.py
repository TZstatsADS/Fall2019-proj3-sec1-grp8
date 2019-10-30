
import pandas as pd
import numpy as np
from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
import warnings

class FiducialDataProcess(object):

	def __init__(self, path, num_data, num_features):
		self.path = path # 
		self.num_data = num_data
		self.num_features = num_features
		self.feature_data = np.zeros((num_data, num_features*(num_features-1)))#np.zeros((2500,78*77))

	def euc_dist(self, p1, p2):
		dist = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
		return int(dist)


	def printkeys(self,i):
		data = loadmat(self.path + '{0:04}'.format(i) +  '.mat')
		return data

	def preprocess(self):

		total_features = []
		for i in range(1, self.num_data+1):
			curr_feature = []
			data = loadmat(self.path + '{0:04}'.format(i) +  '.mat')

			if 'faceCoordinatesUnwarped' in data:
				arr = data['faceCoordinatesUnwarped']
			else:
				arr = data['faceCoordinates2']

			for j in range(arr.shape[0]-1):
				for k in range(j+1, arr.shape[0]):
					curr_feature.append(self.euc_dist(arr[j],arr[k]))
			total_features.append(curr_feature)
		return total_features

	def return_features(self):
		t = self.preprocess()
		return t



class NMF_Data(object):
	def __init__(self, dat_x, dat_y):
		self.dat_x = dat_x
		self.dat_y = dat_y
		self.nmf_features = []
		self.nmf = None

	def create_nmf(self, reduc_comp=100, test_size=500):
		x_train, x_test, y_train, y_test = train_test_split(self.dat_x, self.dat_y, random_state=1, test_size = test_size)
		print(x_train.shape, x_test.shape)
		
		self.nmf = NMF(n_components=reduc_comp, random_state=0)
		self.nmf.fit(x_train)

		x_train_nmf = self.nmf.transform(x_train)
		x_test_nmf = self.nmf.transform(x_test)

		self.nmf_features.append(x_train_nmf)
		self.nmf_features.append(y_train)
		self.nmf_features.append(x_test_nmf)
		self.nmf_features.append(y_test)

	def nmf_dim_reduc(self, data):
		return self.nmf.transform(data)

	def get_nmf_features(self):
		return self.nmf_features

	def save_nmf(self, filename):
		self.create_nmf()
		np.save(filename, self.nmf_features)