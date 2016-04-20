def predict (self , X):
	minDist = np.finfo('float').max
	minClass = -1
	Q = project ( self .W , X. reshape (1 , -1) , self . mu )
	for i in xrange ( len ( self . projections )):
		dist = self . dist_metric ( self . projections [ i], Q)
	if dist < minDist :
		minDist = dist
		minClass = self . y[i]
		return minClass
