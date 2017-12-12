import numpy as np
import csv
from sklearn.svm import SVC
from scipy.optimize import minimize

traindata =[]
testdata=[]
y1=[]
y=[]

#f = open("ionosphere_train.csv")
f = open(raw_input("Enter traindata Filename: "),'r')
traindatafile = csv.reader(f)
#f1 = open("ionosphere_test.csv")
f1 = open(raw_input("Enter testdata Filename: "),'r')
testdatafile = csv.reader(f1)

for row in traindatafile:
	traindata.append(row)
for row in testdatafile:
	testdata.append(row)

for i in range(0,len(traindata)):
	if(traindata[i][len(traindata[0])-1]==" b"):
		y.append(1)
	else:
		y.append(0)

for i in range(0,len(testdata)):
	if(testdata[i][len(testdata[0])-1]==" b"):
		y1.append(1)
	else:
		y1.append(0)

traindata = np.asarray(traindata)
testdata  = np.asarray(testdata)
y  = np.asarray(y)
y1 = np.asarray(y1)

traindata = traindata[:,:-1]
testdata  = testdata[:,:-1]
traindata = np.array(traindata,dtype=float)
testdata  = np.array(testdata,dtype=float)
for i in range(0,len(traindata[0])):
	traindata[:,i] = (traindata[:,i] - min(traindata[:,i]))/(max(traindata[:,i]) - min(traindata[:,i]))
	testdata[:,i] = (testdata[:,i] - min(testdata[:,i]))/(max(testdata[:,i]) - min(testdata[:,i]))

traindata = np.insert(traindata,0,1,axis = 1)
testdata  = np.insert(testdata,0,1,axis = 1)
#print "traindata = ",traindata
#print "Y= ",y
w_old = np.array([1 for i in range(0,len(traindata[0]))])

def func(w):
	w1 = np.asarray(w)
	w2 = np.linalg.norm(w1)**2
	return 1.0

dfdw=[]

def func_derivative(w):
	for i in range(0,len(w)):
		if(i==0):
			dfdw.append(0)
		else:
			dfdw.append(2*w[i])
	dfdw[i] = float(dfdw[i])

	return 10

for i in range(0,len(traindata)):
	if(i==0):
		cons= ({'type' : 'ineq',
		 	 'fun' : lambda w: np.array([y[i]*sum(np.multiply(traindata[i],w))-1]),
		 	 'jac' : lambda w: np.array([np.multiply(y[i],traindata[i])])},)
	else:
		cons+=	({'type' : 'ineq',
		 	 'fun' : lambda w: np.array([y[i]*sum(np.multiply(traindata[i],w))-1]),
		 	 'jac' : lambda w: np.array([np.multiply(y[i],traindata[i])])},)
	

clf=SVC()
clf.fit(traindata,y)
y_pred = clf.predict(traindata)
y_pred_test = clf.predict(testdata)
#res = minimize(func,w_old,jac=func_derivative,constraints = cons,method='SLSQP')
#print ("Output w: ",res.w)
def dataset_error(y_pred,y):
	error_count = 0	
	
	for i in range(0,len(y)):
		if(y_pred[i] != y[i]):
			error_count+= 1

	#print "y = :",y_pred
	return error_count

train_error = dataset_error(y_pred,y)
test_error  = dataset_error(y_pred_test,y1)
#print "w= :",w_old
print "trainerror= :",train_error
print "testerror = :",test_error
#print " y_pred_test",y_pred_test
#print "y1",y1