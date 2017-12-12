import numpy as np
import math

d=10
T = 100
delta = 0.1
c = np.array([0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9,2.1])
eta = c*(math.sqrt(2*(np.log10(d))/T))
regret = np.zeros(c.size)
v = np.zeros(10)

for runs in range(0,50):

	for eta_count in range(0,eta.size):

		w = np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
		for i in range(0,T):
			zt = np.sum(w)
			w = w/zt
			for j in range(0,8):
				v[j] = np.random.binomial(1,0.5)

			v[8] = np.random.binomial(1,0.5 - delta)
			if(i<(T/2)):
				v[9] = np.random.binomial(1,0.5 + delta)
			else:
				v[9] = np.random.binomial(1,0.5 - (2*delta))
		
			regret[eta_count] += np.dot(w,v) - np.amin(v)
			for k in range(0,10):
				w[k] = w[k] * (np.exp(-eta[eta_count]*v[k]))


	regret = regret/50

print "The regret values are",regret





