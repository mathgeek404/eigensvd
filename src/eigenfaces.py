import numpy
from PIL import Image
import os, re




def main():
    datadir = "data/yalefaces/"
    imgList = os.listdir(datadir)
    numArr = []
    fileRegex = re.compile("subject.*")
    for fname in imgList:
        if (fileRegex.match(fname)):
            img = Image.open(datadir+fname)
            arr = numpy.array(img.getdata())
            numArr.append(arr)
            
    
 
    # Joing into matrix
    matrix = numArr[0]
    for i in range(1,len(numArr)-1):
        matrix = numpy.vstack((matrix, numArr[i]))
    matrix = numpy.transpose(matrix)

    #Remove mean
    mean = matrix.mean(axis=1)
    matrix = matrix - mean[:, numpy.newaxis]
    covar = numpy.cov(matrix)


if __name__ == '__main__':
    main()
