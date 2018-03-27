
# coding: utf-8

# In[71]:


import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np


no_time_steps = 28
input_size = 28
hidden_size = 30
output_size = 10
batch_size = 2
num_epochs = 1
learning_rate = 0.01
dtype = torch.DoubleTensor


# MNIST Dataset
train_dataset = dsets.MNIST(root='./data/',
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data/',
                           train=False, 
                           transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

class RNN(torch.nn.Module):
    def __init__(self,input_size,hidden_size,output_size,batch_size):
        super(RNN, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.wxh=Variable(torch.randn(input_size,hidden_size).type(dtype)*0.1,requires_grad=True)
        self.whh=Variable(torch.randn(hidden_size,hidden_size).type(dtype)*0.1,requires_grad=True)
        self.why=Variable(torch.randn(hidden_size,output_size).type(dtype)*0.1,requires_grad=True)
        self.by=Variable(torch.Tensor(batch_size,output_size).type(dtype).zero_(),requires_grad=True)
        self.bh=Variable(torch.Tensor(batch_size,hidden_size).type(dtype).zero_(),requires_grad=True)

        self.mWxh= torch.zeros_like(self.wxh)
        self.mWhh= torch.zeros_like(self.whh)
        self.mWhy= torch.zeros_like(self.why)
        self.mbh= torch.zeros_like(self.bh)
        self.mby= torch.zeros_like(self.by)
        self.dwxh, self.dwhh, self.dwhy = torch.zeros_like(self.wxh), torch.zeros_like(self.whh), torch.zeros_like(self.why)
        self.dbh, self.dby = torch.zeros_like(self.bh), torch.zeros_like(self.by)

    def hidden_init(self,batch_size):
        self.hidden={}
        self.hidden[0]=Variable(torch.Tensor(batch_size,hidden_size).type(dtype).zero_())

    def tanh(self,value):
        return (torch.exp(value)-torch.exp(-value))/(torch.exp(value)+torch.exp(-value))

    def parameter(self):
        self.params = torch.nn.ParameterList([torch.nn.Parameter(self.wxh.data),torch.nn.Parameter(self.whh.data),torch.nn.Parameter(self.why.data),torch.nn.Parameter(self.bh.data),torch.nn.Parameter(self.by.data)])
        return self.params

    def grad_data(self):
        print(self.dwxh,self.dwhy)

    def softmax(self,value):
        return torch.exp(value) / torch.sum(torch.exp(value))

    def updatess(self,lr):
        for param, dparam, mem in zip([self.wxh, self.whh, self.why, self.bh, self.by],
                                [self.dwxh,self.dwhh,self.dwhy,self.dbh,self.dby],
                                [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]):
            mem.data += dparam.data * dparam.data
            param.data += -learning_rate * dparam.data / torch.sqrt(mem.data + 1e-8)                                                                                                                


    def forward(self,inputs,batch_size,no_time_steps,labels):
        self.hidden_init(batch_size)

        inputs=Variable(inputs.type(dtype))
        self.output=Variable(torch.Tensor(no_time_steps,batch_size,self.output_size).type(dtype))

        for t in xrange(no_time_steps):
            if t==0:
                self.hidden[t]=torch.matmul(self.hidden[0],self.whh)
                #print 'time  ',t#,"Inputs",inputs[:,t,:],"Weights",self.wxh
                #print "hidden MATRIX",inputs[:,t,:]
                self.hidden[t]+=torch.matmul(inputs[:,t,:],self.wxh)
                self.hidden[t]=self.tanh(self.hidden[t]+self.bh)
                print 'time  ',t,"Inputs",inputs[:,t,:],"Weights",self.wxh
                #print "HIDDEN MATRIX",self.hidden[t]
            else:
                self.hidden[t]=torch.matmul(self.hidden[t-1],self.whh)#+torch.matmul(self.hidden[t-1],self.whh) 
                #print 'time  ',t#,"Inputs",inputs[:,t,:],"Weights",self.wxh
                self.hidden[t]+=torch.matmul(inputs[:,t,:],self.wxh)
                self.hidden[t]=self.tanh(self.hidden[t]+self.bh)
            print 'time  ',t,"Inputs",inputs[:,t,:],"Weights",self.wxh
            #print "############################################################################################"
            print "hidden MATRIX",self.hidden[t]
        #print t
        self.output[t]=self.softmax(torch.matmul(self.hidden[t],self.why)+self.by)
       # print "OUTPUT MATRIX",self.output[t]            
        return self.output
    def backward(self,loss,label,inputs):
        inputs=Variable(inputs.type(dtype))
        self.dhnext = torch.zeros_like(self.hidden[0])
        self.dy=self.output[27].clone()
        self.dy[:,int(label[0])]=self.dy[:,int(label[0])]-1
            #print(self.dy.shape)
        self.dwhy += torch.matmul( self.hidden[27].t(),self.dy)
        self.dby += self.dy        
        for t in reversed(xrange(no_time_steps)):
            if t==20:
                break
            self.dh = torch.matmul(self.dy,self.why.t()) + self.dhnext # backprop into h  
            self.dhraw = (1 - self.hidden[t] * self.hidden[t]) * self.dh # backprop through tanh nonlinearity          
            self.dbh += self.dhraw #derivative of hidden bias
            self.dwxh += torch.matmul(inputs[:,t,:].t(),self.dhraw) #derivative of input to hidden layer weight
            self.dwhh += torch.matmul( self.hidden[t-1].t(),self.dhraw) #derivative of hidden layer to hidden layer weight
            self.dhnext = torch.matmul(self.dhraw,self.whh.t())            

rnn=RNN(input_size,hidden_size,output_size,batch_size)
def onehot(values,shape):
    temp=torch.Tensor(shape).zero_()
    for k,j in enumerate(labels):
        temp[k][int(j)]=1
    return Variable(temp)

for epoch in range(1):
    for i, (images, labels) in enumerate(train_loader):
        print "images",images.shape
        images = images.view(-1, no_time_steps, input_size)
        outputs = rnn(images,batch_size,no_time_steps,labels)
        labels = Variable(labels.double())
        #print labels,outputs[27,:,:]
        output=outputs[27,:,:]
        labelss=onehot(labels,output.shape)
        #print "labelss",labelss
        loss=-torch.mul(torch.log(output),labelss.double())
        #print "loss",torch.log(output),loss
        loss=torch.sum(loss)
        #print(labels)
        rnn.backward(loss,labels,images)
        rnn.updatess(0.03)
        if i==0:
            break
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))


# In[19]:


# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = images.view(-1, no_time_steps, input_size)
    outputs = rnn.forward(images,batch_size,no_time_steps,labels)
    output=outputs[-1,:,:]
    #print output.data
    _, predicted = torch.max(output.data, 1)
    total += labels.size(0)
    #print labels[0],predicted[0]
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total)) 

# Save the Model
torch.save(rnn.state_dict(), 'rnn.pkl')


# In[41]:


import pandas as pd
no_time_steps = 4
input_size = 5
hidden_size = 3
#num_layers = 2
output_size = 5
batch_size = 1
num_epochs = 2
learning_rate = 0.01

 
def encode(pattern, n_unique):
    encoded = list()
    for value in pattern:
        row = [0.0 for x in range(n_unique)]
        row[value] = 1.0
        encoded.append(row)
    return encoded
 
# create input/output pairs of encoded vectors, returns X, y
def to_xy_pairs(encoded):
	X,y = list(),list()
	for i in range(1, len(encoded)):
		X.append(encoded[i-1])
		y.append(encoded[i])
	return X, y

# convert sequence to x/y pairs ready for use with an LSTM
def to_lstm_dataset(sequence, n_unique):
	# one hot encode
	encoded = encode(sequence, n_unique)
	# convert to in/out patterns
	X,y = to_xy_pairs(encoded)
	# convert to LSTM friendly format
	dfX, dfy = pd.DataFrame(X), pd.DataFrame(y)
	lstmX = dfX.values
	lstmX = lstmX.reshape(1,lstmX.shape[0], lstmX.shape[1])
	lstmY = dfy.values
	return lstmX, lstmY

seq1 = [0,1,2,3,4]
seqYY=[1,2,3,4,0]
seqYY=pd.DataFrame(seqYY).values
seqYY=Variable(torch.from_numpy(seqYY).float())
n_unique = len(set(seq1))
#print n_unique
seq1X, seq1Y = to_lstm_dataset(seq1, n_unique)
#print(seq1X.shape,seq1Y.shape)
inputs=torch.from_numpy(seq1X).float()
labels=torch.from_numpy(seq1Y).float()
print "dsdas",seqYY,"sa"

class RNN(torch.nn.Module):
            def __init__(self,input_size,hidden_size,output_size,batch_size):
                super(RNN, self).__init__()
                self.input_size=input_size
                self.hidden_size=hidden_size
                self.output_size=output_size
                self.wxh=Variable(torch.randn(input_size,hidden_size).type(dtype)*0.1,requires_grad=True)
                self.whh=Variable(torch.randn(hidden_size,hidden_size).type(dtype)*0.1,requires_grad=True)
                self.why=Variable(torch.randn(hidden_size,output_size).type(dtype)*0.1,requires_grad=True)
                self.by=Variable(torch.Tensor(batch_size,output_size).type(dtype).zero_(),requires_grad=True)
                self.bh=Variable(torch.Tensor(batch_size,hidden_size).type(dtype).zero_(),requires_grad=True)

                self.mWxh= torch.zeros_like(self.wxh)
                self.mWhh= torch.zeros_like(self.whh)
                self.mWhy= torch.zeros_like(self.why)
                self.mbh= torch.zeros_like(self.bh)
                self.mby= torch.zeros_like(self.by)
                self.dwxh, self.dwhh, self.dwhy = torch.zeros_like(self.wxh), torch.zeros_like(self.whh), torch.zeros_like(self.why)
                self.dbh, self.dby = torch.zeros_like(self.bh), torch.zeros_like(self.by)

            def hidden_init(self,batch_size):
                self.hidden={}
                self.hidden[0]=Variable(torch.Tensor(batch_size,hidden_size).type(dtype).zero_())

            def tanh(self,value):
                return (torch.exp(value)-torch.exp(-value))/(torch.exp(value)+torch.exp(-value))

            def parameter(self):
                self.params = torch.nn.ParameterList([torch.nn.Parameter(self.wxh.data),torch.nn.Parameter(self.whh.data),torch.nn.Parameter(self.why.data),torch.nn.Parameter(self.bh.data),torch.nn.Parameter(self.by.data)])
                return self.params

            def grad_data(self):
                print(self.dwxh,self.dwhy)

            def softmax(self,value):
                return torch.exp(value) / torch.sum(torch.exp(value))

            def updatess(self,lr):
                for param, dparam, mem in zip([self.wxh, self.whh, self.why, self.bh, self.by],
                                        [self.dwxh,self.dwhh,self.dwhy,self.dbh,self.dby],
                                        [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]):
                    mem.data += dparam.data * dparam.data
                    param.data += -learning_rate * dparam.data / torch.sqrt(mem.data + 1e-8)                                                                                                                


            def forward(self,inputs,batch_size,no_time_steps,labels):
                self.hidden_init(batch_size)

                inputs=Variable(inputs.type(dtype))
                self.output=Variable(torch.Tensor(no_time_steps,batch_size,self.output_size).type(dtype))

                for t in xrange(no_time_steps):
                    if t==0:
                        self.hidden[t]=torch.matmul(self.hidden[0],self.whh)
                        #print 'time  ',t#,"Inputs",inputs[:,t,:],"Weights",self.wxh
                        #print "hidden MATRIX",inputs[:,t,:]
                        self.hidden[t]+=torch.matmul(inputs[:,t,:],self.wxh)
                        self.hidden[t]=self.tanh(self.hidden[t]+self.bh)
                        #print 'time  ',t#,"Inputs",inputs[:,t,:],"Weights",self.wxh
                        #print "HIDDEN MATRIX",self.hidden[t]
                    else:
                        self.hidden[t]=torch.matmul(self.hidden[t-1],self.whh)#+torch.matmul(self.hidden[t-1],self.whh) 
                        #print 'time  ',t#,"Inputs",inputs[:,t,:],"Weights",self.wxh
                        self.hidden[t]+=torch.matmul(inputs[:,t,:],self.wxh)
                        self.hidden[t]=self.tanh(self.hidden[t]+self.bh)
                    #print 'time  ',t#,"Inputs",inputs[:,t,:],"Weights",self.wxh
                    #print "############################################################################################"
                    #print "hidden MATRIX",self.hidden[t]
                #print t
                self.output[t]=self.softmax(torch.matmul(self.hidden[t],self.why)+self.by)
               # print "OUTPUT MATRIX",self.output[t]            
                return self.output
            def backward(self,loss,label,inputs):
                inputs=Variable(inputs.type(dtype))
                print inputs.shape
                self.dhnext = torch.zeros_like(self.hidden[0])
                self.dy=self.output[-1].clone()
                    #print(self.dy.shape)
                self.dy[:,int(label[-1][0])]=self.dy[:,int(label[0])]-1
                    #print(self.dy.shape)
                self.dwhy += torch.matmul( self.hidden[3].t(),self.dy)
                self.dby += self.dy        
                for t in reversed(xrange(no_time_steps)):
                    if t==1:
                        break
                    self.dh = torch.matmul(self.dy,self.why.t()) + self.dhnext # backprop into h  
                    self.dhraw = (1 - self.hidden[t] * self.hidden[t]) * self.dh # backprop through tanh nonlinearity          
                    self.dbh += self.dhraw #derivative of hidden bias
                    self.dwxh += torch.matmul(inputs[:,t,:].t(),self.dhraw) #derivative of input to hidden layer weight
                    self.dwhh += torch.matmul( self.hidden[t-1].t(),self.dhraw) #derivative of hidden layer to hidden layer weight
                    self.dhnext = torch.matmul(self.dhraw,self.whh.t())            

rnn=RNN(input_size,hidden_size,output_size,batch_size)
def onehot(values,shape):
    temp=torch.Tensor(shape).zero_()
    for k,j in enumerate(labels):
            temp[k][int(j)]=1
    return Variable(temp)
labels = Variable(labels.double())

for epoch in range(5):
                outputs = rnn(inputs,batch_size,no_time_steps,labels)
                #print outputs.shape
                output=outputs[-1,:,:]
                #print labels
                #print output
                loss=-torch.mul(torch.log(output),labels.double())
                #print loss
                loss=torch.sum(loss)
                #print(labels)
                rnn.backward(loss,seqYY,inputs)
                rnn.updatess(0.01)
                print(loss.data[0])
                #if i==0:
                 #   break
                #if (i+1) % 100 == 0:
                 #   print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                  #         %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))





# In[449]:


seq1t = [0,1,2,3,4]
n_uniquet = len(set(seq1t))
#print n_unique
seq1Xt, seq1Yt = to_lstm_dataset(seq1t, n_uniquet)
print(seq1Xt.shape,seq1Yt.shape)
seq1Xt=Variable(torch.from_numpy(seq1Xt).float())
labelst=Variable(torch.from_numpy(seq1Yt).float())
output = rnn.forward(seq1Xt,1,4)
output=output.squeeze()
print output,labelst


# In[394]:


for param in rnn.parameter():
    print(type(param.data), param.size())
par=list(rnn.parameter())
print par[0].grad,par[1].grad,par[2].grad


# In[570]:


loss = torch.nn.MSELoss()
input = Variable(torch.randn(3, 5), requires_grad=True)
target = Variable(torch.randn(3, 5))
output = loss(input, target)
output.shape


# In[63]:


x=[1]
y=[2,3,4]
a=(x,y)
print a

