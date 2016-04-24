# Eigenface decomposition codes
# By Sahit Mandala and Will Mitchell

import numpy as np
from PIL import Image
import os, re
from scipy.linalg import svd

class eigRec(object):
    
    def __init__(self, matrix, classVec, imgsize):
        #PCA/Eigendecomp
        u,s,v = svd(matrix, full_matrices=0)
        
        #Choose the k largest eigenvalues, vectors
        k=10
        s[k:] = 0.0
        
        self.Wpca = u[]
		self.mu = mu
		self.matrix = matrix
		self.numimg = numimg
		self.imgsize = imgsize
        self.k = k

		self.proj = [self.project(matrix[:,i]) for i in range(0,numimg)]
		self.projClass = classVec
    
    def eigenReconstruct():
        #Eigenface construction
        for i in range(0,self.k):
            img = Image.fromarray((u[:,i]*s[i]).reshape(imgsize[1],imgsize[0]))
            img.save('result/eigenface%.4i.gif'%i)

    def reconstructImages():    
        #Original image reconstruction
        for kk in range(0,self.numimg):
            iv = np.dot(u,np.dot(np.diag(s),v[:,kk]))
            img = Image.fromarray(iv.reshape(imgsize[1],imgsize[0]))
            img.save('result/recon%.4i.gif'%kk)
        


	def project(self, X):
		return np.dot(self.Wpca, X-self.mu)

	def predict(self , X):
		minDist = np.finfo('float').max
		minClass = -1
		Q = self.project(X)
		for i in range(0,len(self.proj)):
			dist = np.linalg.norm(self.proj[i] - Q)
			if dist < minDist :
				minDist = dist
				minClass = self.projClass[i]
		return minClass

	def reconstruct(self, X):
		return np.dot(self.Wlda.T,X) + self.mu
    
    

    
    
