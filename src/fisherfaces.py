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
mu = matrix.mean(axis=1)
u,s,v = svd((matrix.T-mu).T, full_matrices=0)

#Choose the k largest eigenvalues, vectors
N = len(numArr)
labels = set(classVec)
c = len(labels)

# Projection onto c-dim subspace
W_pca = u[:,0:N-c].T

P = np.dot(W_pca,(matrix.T-mu).T)

print(P.shape)

S_b = np.zeros((P.shape[0],P.shape[0]))
mean  = np.mean(P,axis=1)
for clas in labels:
	indices = [i for i, x in enumerate(classVec) if x == clas]
	N = len(indices)
	mean_i = np.mean(P[:,indices],axis=1)
	S_b = S_b + N*np.dot((mean_i-mean),(mean_i-mean).T)


S_w = np.zeros((P.shape[0],P.shape[0]))
for clas in labels:
	Xi = P[:,np.where(np.asarray(classVec) == clas)[0]]
        indices = [i for i, x in enumerate(classVec) if x == clas]
        mean = np.mean(P[:,indices],axis=1)
        S_w = S_w + np.dot((Xi.T-mean).T,(Xi.T-mean))
       # for i in indices:
        #    S_w = S_w + np.dot((P[:,i]-mean),(P[:,i]-mean).T)
print(S_w)

eigenvalues , eigenvectors = np.linalg.eig(np.linalg.inv(S_w)*S_b)
idx = np.argsort(-eigenvalues)
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]
eigenvalues = np.array(eigenvalues[0:c].real , dtype=np.float32, copy=True )
eigenvectors = np.array(eigenvectors[0:,0:c].real , dtype =np.float32, copy = True)

Wlda = np.dot(eigenvectors.T, W_pca)
print(Wlda.shape)
#Eigenface construction
#for i in range(0,k):
#    img = Image.fromarray((u[:,i]*s[i]).reshape(imgsize[1],imgsize[0]))
#    img.save('result/eigenface%.4i.gif'%i)

def project(W, X, mu = None ):
	if mu is None :
		return np.dot(W, X)
	return np.dot(W, X-mu)

def reconstruct(W, Y ,mu = None ) :
	if mu is None:
		return np.dot(W.T,Y)
	return np.dot(W.T,Y) + mu

#Original image reconstruction
for kk in range(0,numimg):
	iv = matrix[:,kk]
	iv = project(Wlda, iv, mu)
	iv = reconstruct(Wlda, iv, mu)
	img = Image.fromarray(iv.T.reshape(imgsize[1],imgsize[0]))
	img.save('result/lda_recon%.4i.gif'%kk)
