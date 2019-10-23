#import tensorflow as tf
import pandas as pd
import numpy as np
import ipdb
from scipy.io import loadmat


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



if __name__ == '__main__':

	path = 'train_set/points/'
	num_features = 78
	num_data = 2500
	feature_array = FiducialDataProcess(path, num_data, num_features)
	final_features = np.array(feature_array.return_features())
	labelfile = pd.read_csv('train_set/label.csv')
	labels = np.array(labelfile['emotion_idx'])  
	labels2 = np.array(labelfile['type']) 
	for i in range(len(labels2)):
		if labels2[i] == 'simple':
			labels2[i] = 0 
		elif labels2[i] == 'compound':
			labels2[i] = 1

	data_array = np.column_stack((labels2, labels, final_features))
	np.savetxt("dense_data_type_and_emot.csv", data_array, fmt="%d", delimiter=",")







	# for i in range(1,num_data):
	# 	data = feature_array.printkeys(i)
	# 	print(data.keys())







#data = pd.read_csv('fiducial_points.csv',usecols= [i for i in range(2,80)])
