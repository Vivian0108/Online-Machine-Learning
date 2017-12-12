import numpy as np
import math
import random

K=10
T = 10000
delta = 0.1
c = np.array([0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9,2.1])
eta = c*(math.sqrt(2*(np.log10(K))/(K*T)))
#regret = np.zeros(c.size)
v = np.zeros(10)
p = np.zeros(10)
agg_cost = np.zeros(eta.size,K)

for runs in range(0,50):

	for eta_count in range(0,eta.size):

		p = np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
		for i in range(0,T):
			zt = np.sum(p)
			p = p/zt
			#p = ((1-eta[eta_count])*w) + (eta[eta_count]/10) 
			rand_pick = random.random()
			p_cum = np.cumsum(p)
			for l in range(0,K):
				if(rand_pick <=p_cum[l]):
					arm = l
					break

			v = np.zeros(K)
			if(arm < K-2):
				v[arm] = np.random.binomial(1,0.5)/p[arm]
			else if (arm == K-2):
				v[arm] = np.random.binomial(1,0.5 - delta)/p[arm]
			else if (arm == K-1):

				if(i<(T/2)):
					v[arm] = np.random.binomial(1,0.5 + delta)/p[arm]
				else:
					v[arm] = np.random.binomial(1,0.5 - (2*delta))/p[arm]
		
			agg_cost[eta_count] += v
			V_exp = np.exp(-agg_cost[eta_count]*eta[eta_count])
			for k in range(0,10):
				p[k] = (1 - eta[eta_count])*V_exp[j]/np.sum(V_exp) + eta[eta_count]/K;


	regret = regret/50

print "The regret values are",regret





