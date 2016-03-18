import numpy as np
from PIL import Image
import os, re
from scipy.linalg import svd

def pca(X) :
	[n , d] = X.shape

	#Remove mean
	mu = X.mean(axis =0)
	X = X-mu
	
	#Cov matrix
	C = np.cov(X.T)
    
	[ eigenvalues , eigenvectors ] = np.linalg.eigh(C)
		
	idx = np.argsort(-eigenvalues)
	eigenvalues = eigenvalues[ idx ]
	eigenvectors = eigenvectors[:, idx]
	
	return [eigenvalues,eigenvectors,mu]
	

if 1:
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
    matrix = numArr[0][...,np.newaxis]
    for i in range(1,len(numArr)):
        matrix = np.dstack((matrix, numArr[i][...,np.newaxis]))
    #matrix = np.array(numArr).reshape(numArr[0].size,len(numArr))
    g1 = matrix.reshape(243*320,165)
    #matrix = np.array(numArr).reshape([77760,165])
    u,s,v = svd(g1, full_matrices=0)

    s[10:] = 0.0


    # [eigval, eigvec, mu] = pca(matrix)
    
    #for column in array.T:
    for kk in range(165):
        iv = np.dot(u,np.dot(np.diag(s),v[:,kk]))
        #img = Image.fromarray(numArr[0])
        img = Image.fromarray(iv.reshape(numArr[0].shape))
        img.save('out%.4i.gif'%kk)



#if __name__ == '__main__':
#    main()
