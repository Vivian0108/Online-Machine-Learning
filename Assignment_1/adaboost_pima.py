import numpy as np
import csv

traindata =[]
testdata=[]
y1=[]
y=[]
num_iterations = 100

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
		y.append(-1)

for i in range(0,len(testdata)):
	if(testdata[i][len(testdata[0])-1]=="tested_positive"):
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

# input: dataset X and labels y (in {+1, -1})
hypotheses = []
hypothesis_weights = []
hypothesis_signs = []

N= traindata.shape[0]
#print "N",N
d = np.ones(N) / N

def weighted_dataset_error(y_pred,y,d):
	error_count = 0	
	
	for i in range(0,len(y)):
		if(y_pred[i] != y[i]):
			error_count+= 1*d[i]

	#print "y = :",y_pred
	
	return error_count

def dataset_error(y_pred,y):
	error_count = 0	
	
	for i in range(0,len(y)):
		if(y_pred[i] != y[i]):
			error_count+= 1

	#print "y = :",y_pred
	
	return error_count

ht_pred=np.zeros([len(traindata),len(traindata[0])-1])
ht_pred1=np.zeros([len(traindata),len(traindata[0])-1])
#print "traindata",traindata
count = 0

for t in range(num_iterations):
        
	pred=[]    
	error=[]   
	error1=[]
	tot_err =[]

	for i in range(0,len(traindata[0])-1):
		mean = np.sum(np.multiply(d,traindata[:,i].T))  
		#print "mean",mean  	
		for j in range(0,len(traindata)):
			if(traindata[j,i]<mean):
				ht_pred[j][i] = -1
				ht_pred1[j][i] = 1
			else:
				ht_pred[j][i] = 1
				ht_pred1[j][i] = -1
    
		error.append(weighted_dataset_error(ht_pred[:,i].T,y,d))		
		error1.append(weighted_dataset_error(ht_pred1[:,i].T,y,d))

	#print "error",error
	#print "error1",error1
	
	if(min(error)<min(error1)):
   		dim = error.index(np.min(error))
   		b=1
	if(min(error)>=min(error1)):
   		dim = error1.index(np.min(error1))
   		b=-1
   	
	#count = count + 1
	#print "count",count 	
	#print "dim",dim
	mean_dim = np.mean(traindata[:,dim])
	for j in range(0,len(traindata)):
  		if(traindata[j,dim]<mean_dim):
  			pred.append(-1*b)
  		else:
  			pred.append(1*b) 

	err=[]
    
	for i in range(0,len(y)):
   		if(pred[i]!=y[i]):
			err.append(1)
		else:
			err.append(0)

	#print "d",d
	eps = d.dot(err)
	#print"eps",eps

	alpha = (np.log(1 - eps) - np.log(eps)) / 2
    #print "alpha",alpha
	d = d * np.exp([-1*alpha*y_elem*p for y_elem,p in zip(y,pred)])
	d = d / d.sum()

	hypotheses.append(dim)
	hypothesis_weights.append(alpha)
	hypothesis_signs.append(b)

def predict(data,dim,b):
	pred2=[]
	mean_dim = np.mean(data[:,dim])
	for j in range(0,len(data)):
  		if(data[j,dim]<mean_dim):
  			pred2.append(-1*b)
  		else:
  			pred2.append(1*b)
  	return pred2


y_pred_train = np.zeros(N)
for (dim, alpha,b) in zip(hypotheses, hypothesis_weights,hypothesis_signs):
	#print"dim,alpha,b",dim,alpha,b
	pre = predict(traindata,dim,b)
	y_pred_train = y_pred_train + [alpha*pre_var for pre_var in pre]

y_pred_train = np.sign(y_pred_train)
train_error = dataset_error(y_pred_train,y)
print "trainerror= :",train_error


y_pred_test = np.zeros(len(testdata))
for (dim, alpha,b) in zip(hypotheses, hypothesis_weights,hypothesis_signs):
	#print"dim,alpha,b",dim,alpha,b
	pre1 = predict(testdata,dim,b)
	y_pred_test = y_pred_test + [alpha*pre_var1 for pre_var1 in pre1]

y_pred_test = np.sign(y_pred_test)
test_error = dataset_error(y_pred_test,y1)
print "testerror = ",test_error


