import numpy as np
import csv

traindata =[]
testdata=[]
y1=[]
y=[]
num_iters = 2000

f = open(raw_input("Enter traindata Filename: "),'r')
#f = open("ionosphere_train.csv")
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
		y.append(-1)

for i in range(0,len(testdata)):
	if(testdata[i][len(traindata[0])-1]==" b"):
		y1.append(1)
	else:
		y1.append(-1)
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
w_old = np.ones(len(traindata[0]))
w_new = np.ones(len(traindata[0]))
for t in range(num_iters):
	for i in range(0,len(traindata)):
		if(y[i]*(np.dot(w_old,traindata[i,:]))<=0):
			w_new = w_old + np.multiply(0.05*y[i],traindata[i,:])

	w_old = w_new

def dataset_error(y_pred,y):
	error_count = 0	
	
	for i in range(0,len(y)):
		if(y_pred[i] != y[i]):
			error_count+= 1

	#print "y = :",y_pred
	
	return error_count

pred_train=[]
pred_test =[]
for j in range(0,len(traindata)):
		if(np.dot(w_old,traindata[j,:]) < 0):
			pred_train.append(-1)
		else:
			pred_train.append(1)

for j in range(0,len(testdata)):
		if(np.dot(w_old,testdata[j,:]) < 0):
			pred_test.append(-1)
		else:
			pred_test.append(1)

train_error = dataset_error(pred_train,y)
test_error = dataset_error(pred_test,y1)

print "train_error",train_error
print "test_error",test_error
print " Hypothesis w=",w_old





