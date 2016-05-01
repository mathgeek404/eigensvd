# Fisherface decomp codes
# Author: Sahit Mandala
# Notes: http://www.bytefish.de/blog/fisherfaces/

import numpy as np
from PIL import Image
import os, re
from scipy.linalg import svd

class FFRec(object):

	def __init__(self, matrix, classVec, imgsize):

		#PCA/Eigendecomp
		mu = matrix.mean(axis=1)
		u,s,v = svd((matrix.T-mu).T, full_matrices=0)

		#Num of Classes
		N = len(classVec)
		labels = set(classVec)
		c = len(labels)


		# Projection onto N-c-dim subspace
		W_pca = u[:,0:N-c].T
		P = np.dot(W_pca,(matrix.T-mu).T)


		S_b = np.zeros((P.shape[0],P.shape[0]))
		mean  = np.mean(P,axis=1)
		for clas in labels:
			indices = [i for i, x in enumerate(classVec) if x == clas]
			nn = len(indices)
			mean_i = np.mean(P[:,indices],axis=1)
			S_b = S_b + nn*np.dot((mean_i-mean),(mean_i-mean).T)


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
		self.numimg = N
		self.imgsize = imgsize
		self.s = eigenvalues


		self.proj = [self.project(matrix[:,i]) for i in range(0,N)]
		self.projClass = classVec


	def project(self, X):
		return np.dot(self.Wlda, X-self.mu)


	def predict(self , X):
		minDist = np.finfo('float').max
		minClass = -1
		Q = self.project(X)
		for i in range(0,len(self.proj)):
			dist = np.linalg.norm(self.proj[i] - Q)
			#print dist
			if dist < minDist :
				minDist = dist
				minClass = self.projClass[i]
		return minClass

	def reconstruct(self, X):
		return np.dot(self.Wlda.T,X) + self.mu

	def reconstructImages(self):
		#Original image reconstruction
		for kk in range(0,self.numimg):
			iv = self.matrix[:,kk]
			iv = self.project(iv)
			iv = self.reconstruct(iv)
			img = Image.fromarray(self.matrix[:,kk].T.reshape(self.imgsize[1],self.imgsize[0]))
			img.save('result/lda_recon%.4iORIG.gif'%kk)
			img = Image.fromarray(iv.T.reshape(self.imgsize[1],self.imgsize[0]))
			img.save('result/lda_recon%.4i.gif'%kk)
	def eigenReconstruct(self):
		#Eigenface construction
		img = Image.fromarray((self.mu).reshape(self.imgsize[1],self.imgsize[0]))
		img.save('result/fishermeanface.gif')
		for i in range(0,self.Wlda.shape[0]):
			img = Image.fromarray((self.Wlda[i,:].T*self.s[i] + self.mu).reshape(self.imgsize[1],self.imgsize[0]))
			img.save('result/fisherface%.4i.gif'%i)
