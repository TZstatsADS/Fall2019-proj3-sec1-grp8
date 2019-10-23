import numpy as np
import pandas as pd
from scipy.io import loadmat
import ipdb
#images are 1000 x 750 pixels
labelfile = pd.read_csv('train_set/label.csv')
labels = np.array(labelfile['emotion_idx'])  

path = 'train_set/points/'
num_data = 2500

all_sparse_images = []

for i in range(1, num_data+1):
	sparse_img = np.zeros((1000,750))
	data = loadmat(path + '{0:04}'.format(i) +  '.mat')

	if 'faceCoordinatesUnwarped' in data:
		arr = data['faceCoordinatesUnwarped']
	else:
		arr = data['faceCoordinates2']

	for j in range(len(arr)):
		sparse_img[int(arr[j][0]), int(arr[j][1])] = 1
	sparse_img.astype(int)
	all_sparse_images.append(sparse_img)

np.save('sparse_img_np.npy', all_sparse_images)


