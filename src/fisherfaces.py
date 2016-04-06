# Fisherface decomp codes
# Author: Sahit Mandala
# Notes: http://www.bytefish.de/blog/fisherfaces/

import numpy as np
from PIL import Image
import os, re
from scipy.linalg import svd

#Main Scriptss
imgsize=(0,0)
numimg = 0
datadir = "data/yalefaces/"
imgList = os.listdir(datadir)
numArr = []
classVec = []
fileRegex = re.compile("subject.*")
for fname in imgList:
	if (fileRegex.match(fname)):
		classnum = int(filter(str.isdigit, fname))
		classVec.append(classnum)
		img = Image.open(datadir+fname).convert('L')
		arr = np.array(img, 'uint8').reshape(img.size[0]*img.size[1])
		numArr.append(arr)
		imgsize  = img.size
		numimg += 1

# Join into matrix, creating a (num of dim) x (num of data pts) matrix
matrix = np.vstack(numArr).T

print(matrix.shape)
#PCA/Eigendecomp
u,s,v = svd(matrix, full_matrices=0)

#Choose the k largest eigenvalues, vectors
N = len(numArr)
labels = set(classVec)
c = len(labels)

# Projection onto c-dim subspace
P = np.dot(u[:,0:N-c],np.dot(np.diag(s[0:N-c]),v[0:N-c,:]))

S_b = np.zeros((P.shape[0],P.shape[0]))
for clas in labels:
	indices = [i for i, x in enumerate(classVec) if x == clas]
	mean = np.mean(P[:,indices],axis=1)
	for i in indices:
		S_b = S_b + len(indices)*(P[:,i]-mean)*(P[:,i]-mean).T
	
print(S_b)
