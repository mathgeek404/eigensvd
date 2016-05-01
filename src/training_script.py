import numpy as np
from PIL import Image
import os, re
from scipy.linalg import svd
from eigenfaces import eigRec
from fisherfaces import FFRec

#a = FFRec()
#print a.predict(a.matrix[:, 164])



# Yale database
imgsize=(0,0)
numimg = 0
datadir = "data/yalefaces/train/"
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

eigAcc = 0
fishAcc = 0
for i in range(0,0):
    trains = range(0,i)+range(i+1,numimg)
    eig = eigRec(matrix[:,trains], [classVec[tmp] for tmp in trains], imgsize)
    #fish = FFRec(matrix[:,trains], [classVec[tmp] for tmp in trains], imgsize)
    if (eig.predict(numArr[i].T) == classVec[i]):
        eigAcc = eigAcc + 1
    #if (fish.predict(numArr[i].T) == classVec[i]):
    #    fishAcc = fishAcc + 1

print numimg
print eigAcc
print fishAcc


eigAcc = 0
fishAcc = 0

fish = FFRec(matrix, classVec, imgsize)
fish.eigenReconstruct()
fish.reconstructImages()
#TODO: Eigenface reconstruction
#eig.eigenReconstruct()

imgsize=(0,0)
numimg = 0
datadir = "data/yalefaces/test/"
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

eigAcc = 0
fishAcc = 0
#for i in range(0,numimg):
#    if (eig.predict(numArr[i].T) == classVec[i]):
#        eigAcc = eigAcc + 1
#    if (fish.predict(numArr[i].T) == classVec[i]):
#        fishAcc = fishAcc + 1


#print numimg
#print eigAcc
#print fishAcc
'''
print "Caltech data"
# Caltech data
imgsize=(0,0)
numimg = 0
numArr = []
classVec = []
datadir = "data/bird-plane/bird/"
imgList = os.listdir(datadir)
fileRegex = re.compile(".*jpg")
for fname in imgList:
    if (fileRegex.match(fname)):
        classVec.append("bird")
        img = Image.open(datadir+fname).convert('L')
        arr = np.array(img, 'uint8').reshape(img.size[0]*img.size[1])
        numArr.append(arr)
        imgsize  = img.size
        numimg += 1
datadir = "data/bird-plane/llama/"
imgList = os.listdir(datadir)
fileRegex = re.compile(".*jpg")
for fname in imgList:
    if (fileRegex.match(fname)):
        classVec.append("llama")
        img = Image.open(datadir+fname).convert('L')
        arr = np.array(img, 'uint8').reshape(img.size[0]*img.size[1])
        numArr.append(arr)
        imgsize  = img.size
        numimg += 1

matrix = np.vstack(numArr).T

eig = eigRec(matrix, classVec, imgsize)
fish = FFRec(matrix, classVec, imgsize)

imgsize=(0,0)
numimg = 0
numArr = []
classVec = []
datadir = "data/bird-plane/bird-test/"
imgList = os.listdir(datadir)
fileRegex = re.compile(".*jpg")
for fname in imgList:
    if (fileRegex.match(fname)):
        classVec.append("bird")
        img = Image.open(datadir+fname).convert('L')
        arr = np.array(img, 'uint8').reshape(img.size[0]*img.size[1])
        numArr.append(arr)
        imgsize  = img.size
        numimg += 1
datadir = "data/bird-plane/llama-test/"
imgList = os.listdir(datadir)
fileRegex = re.compile(".*jpg")
for fname in imgList:
    if (fileRegex.match(fname)):
        classVec.append("llama")
        img = Image.open(datadir+fname).convert('L')
        arr = np.array(img, 'uint8').reshape(img.size[0]*img.size[1])
        numArr.append(arr)
        imgsize  = img.size
        numimg += 1

eigAcc = 0
fishAcc = 0
for i in range(0,numimg):
    if (eig.predict(numArr[i].T) == classVec[i]):
        eigAcc = eigAcc + 1
    if (fish.predict(numArr[i].T) == classVec[i]):
        fishAcc = fishAcc + 1

print numimg
print eigAcc
print fishAcc
'''
