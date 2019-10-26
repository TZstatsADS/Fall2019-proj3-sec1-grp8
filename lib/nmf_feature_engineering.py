from data_preprocess import FiducialDataProcess
import ipdb
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
from sklearn.externals import joblib

class NMF_Data(object):
	def __init__(self, dat_x, dat_y):
		self.dat_x = dat_x
		self.dat_y = dat_y
		self.nmf_features = []

	def create_nmf(self, reduc_comp=300, test_size=500):
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

	def get_nmf_features(self):
		return self.nmf_features

	def save_nmf(self, filename):
		self.create_nmf()
		np.save(filename, self.nmf_features)

	def save_nmf_model(self, filename):
		joblib.dump(self.nmf, filename)


if __name__ == '__main__':

	my_data = np.genfromtxt('dense_data_type_and_emot.csv', delimiter=',')
	np.random.seed(0)
	np.random.shuffle(my_data)	
	features_init = my_data[:,1:]
	features = features_init / features_init.max(axis=0)
	labels = my_data[:,0:2] 
	nmf_total = NMF_Data(features, labels)
	filename = 'nmf_features_type_emot_300.npy'
	nmf_total.save_nmf(filename)




	# dat = pd.read_csv("nmf_faducial.csv", index_col=0)
	# dat_x = dat.loc[:,'feature1':'feature6006']
	# dat_y = dat.loc[:, 'emotion_idx']
	# nmf_total = NMF_Data(dat_x, dat_y)
	# nmf_total.save_nmf()
