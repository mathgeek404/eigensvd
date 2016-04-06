import numpy as np
from PIL import Image
import os, re
# Import matplotlib

def pca(X) :
	
	[n , d] = X.shape

	#Remove mean
	mu = X.mean(axis =0)
	X = X-mu
	
	u,s,v = svd(matrix)
	
	#Cov matrix
	C = np.cov(X)
    
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
