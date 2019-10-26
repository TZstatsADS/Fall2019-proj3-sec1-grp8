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
#################DATA PREPROCESSING CLASS###################3


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



####################DATA PREPROCESSING SCRIPT ########################## 

path = '../../train_set/points/' #SPECIFY FOLDER PATH WHERE THE NEW FIDUCIAL DATAPOINTS ARE
path_, dirs, files = next(os.walk(path))
file_count = len(files)

num_features = 78
num_data = file_count
feature_array = FiducialDataProcess(path, num_data, num_features)
final_features = np.array(feature_array.return_features())


############################# FEATURE ENGINEERING CLASS###########################################
import pickle

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

#################################### FEATURE ENGINEERING SCRIPT ##########################################

original_data = np.genfromtxt('dense_data_type_and_emot.csv', delimiter=',')
np.random.seed(0)
np.random.shuffle(original_data)	
features_init = original_data[:,2:]
features = features_init / features_init.max(axis=0)
labels = original_data[:,0:2] 

orig_bin_nmf_features = NMF_Data(features, labels)
orig_bin_nmf_features.create_nmf(reduc_comp=100)
new_bin_nmf_features = orig_bin_nmf_features.nmf_dim_reduc(final_features)
x_test_set_bin = new_bin_nmf_features

orig_emot_nmf_features = NMF_Data(features, labels)
orig_emot_nmf_features.create_nmf(reduc_comp=300)
new_emot_nmf_features = orig_emot_nmf_features.nmf_dim_reduc(final_features)
x_test_set_emot = new_emot_nmf_features

test_set_len = len(x_test_set_bin)
#nmf_features  = np.load('nmf_features_type_emot.npy',allow_pickle=True) #for debugging
#nmf_features2  = np.load('nmf_features_type_emot_300.npy',allow_pickle=True) # for debugging


############################ TESTING EXECUTIVE ############################

binary_model = load_model('binary_classification_model.h5',  custom_objects={'leaky_relu': tf.nn.leaky_relu})
binary_predictions = binary_model.predict(x_test_set_bin)
binary_predictions = [binary_predictions[i].argmax() for i in range(len(binary_predictions))]   

compound_model = load_model('compound_classification_model.h5',  custom_objects={'leaky_relu': tf.nn.leaky_relu})
simple_model = load_model('simple_classification_model.h5',  custom_objects={'leaky_relu': tf.nn.leaky_relu})

emotion_dict = {1: 'Neutral', 2: 'Happy', 3: 'Sad', 4: 'Angry', 5:'Surprised',
				6: 'Disgusted', 7:'Fearful', 8:'Happily surprised', 9: 'Happily disgusted',
				10: 'Sadly angry', 11: 'Angrily disgusted', 12: 'Appalled', 13: 'Hatred',
				14:'Angrily surprised', 15:'Sadly surprised', 16:'Disgustedly surprised',
				17:'Fearfully surprised', 18:'Awed', 19:'Sadly fearful', 20:'Fearfully disgusted',
				21:'Fearfully angry', 22:'Sadly disgusted'}


emot_idx = [0]*test_set_len
idx = [i for i in range(test_set_len)]

compound_prediction = compound_model.predict(x_test_set_emot)
simple_prediction = simple_model.predict(x_test_set_emot)

for i in range(len(x_test_set_bin)):
	if binary_predictions[i] == 1:
		emot_idx[i] = compound_prediction[i,:].argmax() + 8
		binary_predictions[i] = 'compound'

	elif binary_predictions[i] == 0:
		emot_idx[i] = simple_prediction[i,:].argmax() + 1
		binary_predictions[i] = 'simple'


df = pd.DataFrame(data={" ": idx, "Index": idx, 
							"identity": [None]*len(x_test_set_bin), "emotion_idx": emot_idx,
							"emotion_cat": [emotion_dict[emot_idx[i]] for i in range(test_set_len)], "type": binary_predictions})

df.to_csv("./Proj3_Section1_Group8_TEST_labels.csv", sep=',',index=False)
#### PLEASE ADD A LINE OF CODE TO DOWNLOAD THIS CSV FILE SO THAT THE TA CAN GET IT!!! 