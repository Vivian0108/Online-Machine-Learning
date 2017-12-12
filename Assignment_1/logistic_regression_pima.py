import numpy as np
import csv

traindata =[]
testdata=[]
y1=[]
y=[]
num_iters = 5000

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
	if(traindata[i][len(traindata[0])-1]=="tested_positive"):
		y.append(1)
	else:
		y.append(0)

for i in range(0,len(testdata)):
	if(testdata[i][len(traindata[0])-1]=="tested_positive"):
		y1.append(1)
	else:
		y1.append(0)
traindata = np.asarray(traindata)
testdata  = np.asarray(testdata)
y  = np.asarray(y)
y1 = np.asarray(y1)
#print "y1 =",y1

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
w_old	  = np.array([1 for i in range(0,len(traindata[0]))])
gradient  = []
n=0

def sigmoid(traindata,w):
	weighted_sum = np.dot(traindata,w.T)

	func_w =np.array([0 for j in range(0,len(weighted_sum))])
	for i in range(0,len(weighted_sum)):
		func_w[i] = 1/(1+np.exp(np.negative(weighted_sum[i])))
	return func_w

while n<num_iters :	
	
	f_w = sigmoid(traindata,w_old)

	for i in range(0,len(traindata)):
		if(f_w[i]<0.5):
			f_w[i] = 0
		else:
			f_w[i] =1

  		if(f_w[i] != y[i]):
  			diff = y[i] - f_w[i]
			w_new = w_old + 0.05*np.multiply(diff,traindata[i,:])
			w_old = w_new	
	#print("n=",n)
	n=n+1

def dataset_error(w,data,y):
	error_count = 0	
	y_pred = sigmoid(data,w)

	for i in range(0,len(data)):
		if(y_pred[i]<0.5):
			y_pred[i] =0
		if(y_pred[i]>=0.5):
			y_pred[i] = 1
		if(y_pred[i] != y[i]):
			error_count+= 1

	#print "y = :",y_pred
	return error_count

train_error = dataset_error(w_old,traindata,y)
test_error  = dataset_error(w_old,testdata,y1)
#print "w= :",w_old
print "trainerror= :",train_error
print "testerror = :",test_error

