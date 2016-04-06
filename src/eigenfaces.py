# Eigenface decomposition codes
# By Sahit Mandala and Will Mitchell

import numpy as np
from PIL import Image
import os, re
from scipy.linalg import svd

def eigendec(X) :
	[n , d] = X.shape

	#Remove mean
	mu = X.mean(axis =0)
	X = X-mu

	u,s,v = svd(matrix)

	#Cov matrix
	C = np.cov(X.T)

	[ eigenvalues , eigenvectors ] = np.linalg.eigh(C)

	idx = np.argsort(-eigenvalues)
	eigenvalues = eigenvalues[ idx ]
	eigenvectors = eigenvectors[:, idx]

	return [eigenvalues,eigenvectors,mu]

	
def main():
    imgsize = 0
    datadir = "data/yalefaces/"
    imgList = os.listdir(datadir)
    numArr = []
    fileRegex = re.compile("subject.*")
    for fname in imgList:
        if (fileRegex.match(fname)):
            img = Image.open(datadir+fname).convert('L')
            arr = np.array(img, 'uint8')
            numArr.append(arr)
            imgsize  = img.size
        
    # Joing into matrix
    matrix = numArr[0].reshape(numArr[0].size)
    for i in range(1,len(numArr)):
        matrix = np.vstack((matrix, numArr[i]))
    #matrix = numpy.transpose(matrix)

	matrix = np.array(numArr).reshape(40095, 320)

    [eigval, eigvec, mu] = pca(matrix)
    
    #for column in array.T:
    #img = Image.fromarray(numArr[0])
    img2.save('out.gif')



if __name__ == '__main__':
    main()



imgsize=(0,0)
numimg = 0
datadir = "data/yalefaces/"
imgList = os.listdir(datadir)
numArr = []
fileRegex = re.compile("subject.*")
for fname in imgList:
	if (fileRegex.match(fname)):
		img = Image.open(datadir+fname).convert('L')
		arr = np.array(img, 'uint8').reshape(img.size[0]*img.size[1])
		numArr.append(arr)
		imgsize  = img.size
		numimg += 1

# Join into matrix, creating a (num of dim) x (num of data pts) matrix
matrix = np.vstack(numArr).T

#PCA/Eigendecomp
u,s,v = svd(matrix, full_matrices=0)

#Choose the k largest eigenvalues, vectors
k=10
s[k:] = 0.0

#Eigenface construction
for i in range(0,k):
    img = Image.fromarray((u[:,i]*s[i]).reshape(imgsize[1],imgsize[0]))
    img.save('result/eigenface%.4i.gif'%i)

#Original image reconstruction
for kk in range(0,numimg):
    iv = np.dot(u,np.dot(np.diag(s),v[:,kk]))
    img = Image.fromarray(iv.reshape(imgsize[1],imgsize[0]))
    img.save('result/recon%.4i.gif'%kk)

