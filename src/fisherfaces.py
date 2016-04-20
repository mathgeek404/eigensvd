# Fisherface decomp codes
# Author: Sahit Mandala
# Notes: http://www.bytefish.de/blog/fisherfaces/

import numpy as np
from PIL import Image
import os, re
from scipy.linalg import svd

class fisherRecognizer(object):

	def __init__(self, datadir ="data/yalefaces/"):
		imgsize=(0,0)
		numimg = 0
		#datadir = "data/yalefaces/"
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

		#PCA/Eigendecomp
		mu = matrix.mean(axis=1)
		u,s,v = svd((matrix.T-mu).T, full_matrices=0)

		#Num of Classes
		N = len(numArr)
		labels = set(classVec)
		c = len(labels)

		# Projection onto c-dim subspace
		W_pca = u[:,0:N-c].T
		P = np.dot(W_pca,(matrix.T-mu).T)

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


		eigenvalues , eigenvectors = np.linalg.eig(np.linalg.inv(S_w)*S_b)
		idx = np.argsort(-eigenvalues)
		eigenvalues = eigenvalues[idx]
		eigenvectors = eigenvectors[:,idx]
		eigenvalues = np.array(eigenvalues[0:c].real , dtype=np.float32, copy=True )
		eigenvectors = np.array(eigenvectors[0:,0:c].real , dtype =np.float32, copy = True)

		self.Wlda = np.dot(eigenvectors.T, W_pca)
		self.mu = mu
		self.matrix = matrix
		self.numimg = numimg
		self.imgsize = imgsize

		#Eigenface construction
		#for i in range(0,k):
		#    img = Image.fromarray((u[:,i]*s[i]).reshape(imgsize[1],imgsize[0]))
		#    img.save('result/eigenface%.4i.gif'%i)

	def project(self, X):
		return np.dot(self.Wlda, X-self.mu)

	def reconstruct(self, Y):
		return np.dot(self.Wlda.T,Y) + self.mu

	def reconstructImages(self):
		#Original image reconstruction
		for kk in range(0,self.numimg):
			iv = self.matrix[:,kk]
			iv = self.project(iv)
			iv = self.reconstruct(iv)
			img = Image.fromarray(iv.T.reshape(self.imgsize[1],self.imgsize[0]))
			img.save('result/lda_recon%.4i.gif'%kk)
