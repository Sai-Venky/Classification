
# coding: utf-8

# In[32]:


from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

# extract features from each photo in the directory
def extract_features(directory):
	# load the model
	model = VGG16()
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# summarize
	print(model.summary())
	# extract features from each photo
	features = dict()
	for name in listdir(directory):
		# load an image from file
		filename = directory + '/' + name
		image = load_img(filename, target_size=(224, 224))
		# convert the image pixels to a numpy array
		image = img_to_array(image)
		# reshape data for the model
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the VGG model
		image = preprocess_input(image)
		# get features
		feature = model.predict(image, verbose=0)
		# get image id
		image_id = name.split('.')[0]
		# store feature
		features[image_id] = feature
		print('>%s' % name)
	return features

# extract features from all images
directory = 'Flicker8k_Dataset'
features = extract_features(directory)
print('Extracted Features: %d' % len(features))
# save to file
dump(features, open('features.pkl', 'wb'))


# In[1]:


from pickle import load

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions

# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features

# load training dataset (6K)
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = load_photo_features('features.pkl', train)
print('Photos: train=%d' % len(train_features))


# In[2]:


from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions

# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features

# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, descriptions, photos):
	X1, X2, y = list(), list(), list()
	# walk through each image identifier
	for key, desc_list in descriptions.items():
		# walk through each description for the image
		for desc in desc_list:
			# encode the sequence
			seq = tokenizer.texts_to_sequences([desc])[0]
			# split one sequence into multiple X,y pairs
			for i in range(1, len(seq)):
				# split into input and output pair
				in_seq, out_seq = seq[:i], seq[i]
				# pad input sequence
				in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
				# encode output sequence
				out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
				# store
				X1.append(photos[key][0])
				X2.append(in_seq)
				y.append(out_seq)
	return array(X1), array(X2), array(y)
# train dataset

# load training dataset (6K)
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = load_photo_features('features.pkl', train)
print('Photos: train=%d' % len(train_features))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)
# prepare sequences
#X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features)

# dev dataset

# load test set
'''filename = 'Flickr8k_text/Flickr_8k.devImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_photo_features('features.pkl', test)
print('Photos: test=%d' % len(test_features))
# prepare sequences
X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features)
'''


# In[3]:


def create_sequences(tokenizer, max_length, descriptions, photos):
	X1, X2, y = list(), list(), list()
	# walk through each image identifier
	for k,[key, desc_list] in enumerate(descriptions.items()):
		if(k==10):
			break#print kwalk through each description for the image
		for desc in desc_list:
			# encode the sequence
			seq = tokenizer.texts_to_sequences([desc])[0]
			# split one sequence into multiple X,y pairs
			for i in range(1, len(seq)):
				# split into input and output pair
				in_seq, out_seq = seq[:i], seq[i]
				# pad input sequence
				in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
				# encode output sequence
				out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
				# store
				X1.append(photos[key][0])
				print len(X1)
				X2.append(in_seq)
				y.append(out_seq)
			
	return array(X1), array(X2), array(y)
X1trainss, X2trainss, ytrainss = create_sequences(tokenizer, max_length, train_descriptions, train_features)


# In[7]:


print ytrainss[1][0:396]


# In[4]:


import torch
import torch.nn as nn
from torch.autograd import Variable


# In[102]:


for k,[key, desc_list] in enumerate(train_descriptions.items()):
    print k,train_features[key][1]


# In[5]:


Xi=torch.from_numpy(X1trainss)
Xj=torch.from_numpy(X2trainss)
Yo=torch.from_numpy(ytrainss)


# In[6]:


Xi=Variable(Xi)


# In[7]:


Xj=Variable(Xj)


# In[8]:


Yo=Variable(Yo)


# In[9]:


Xj=Xj.type(torch.LongTensor)


# In[15]:


oops=Xj[:1]
joops=Xi[:1]
print oops,joops
print Yo[:1]


# In[10]:


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, 256)
        self.dropout = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(256, hidden_size, 1, batch_first=True)
        self.linear = nn.Linear(4096, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3= nn.Linear(256, 7579)
    
    def forward(self,features,captions):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        print "done"
        embeddings = self.dropout(embeddings)
        print "donee"
        lstmout, _ = self.lstm(embeddings)
        print "doneeee"
        features=self.dropout(features)
        features=self.linear(features)
        outputs=lstmout[:,-1,:]+features
        outputs=self.linear2(outputs)
        outputs=self.linear3(outputs)
        return outputs
    


# In[13]:


model=DecoderRNN(256,7579)


# In[54]:


model(joops,oops)


# In[61]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
Xis=Xi[:1]
Xjs=Xj[:1]
Yos=Yo[:1]


# In[1]:


# Train the Model
for epoch in range(50):
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(Xi,Xj)
        loss = criterion(outputs, Yo)
        loss.backward()
        optimizer.step()
        
        print ('Epoch [%d], Loss: %.4f' 
                   %(epoch+1, loss.data[0]))


# In[48]:


for i<10:
    print i
    i=9


# In[1]:


import torch


# In[30]:


a=torch.zeros((3,10))


# In[31]:


a


# In[32]:


a[0][3]=1
a[1][8]=1
a[2][5]=1


# In[33]:


a
m,mn=torch.Tensor,torch.Tensor


# In[34]:


m=torch.max(a,1,keepdim=False)


# In[35]:


m[1]


# In[18]:


feature=torch.randn((1,4096))


# In[19]:


feature


# In[31]:


temp='startseq'
seq=tokenizer.texts_to_sequences([temp])[0]
x_test_desc=pad_sequences([seq], maxlen=max_length)
x_test_desc=torch.from_numpy(x_test_desc)
x_test_desc=x_test_desc.type(torch.LongTensor)
print x_test_desc.shape
feature=Variable(feature)
x_test_desc=Variable(x_test_desc)
output='startseq'


# In[11]:


def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None


# In[36]:


predict = model(feature,x_test_desc)
                                                          #pred={"x1":x[0],"x2":x2[0]}
tmp=word_for_id(i['classes'],tokenizer)
temp+=' '
temp+=tmp
    #print "temp"
    #print temp
seq=tokenizer.texts_to_sequences([temp])[0]
    #print seq
x_test_desc=pad_sequences([seq], maxlen=max_length)[0]
    #print x_test_desc.shape
x_test_desc=x_test_desc.reshape((1,x_test_desc.shape[0]))
    #print x_test_desc.shape
output+=' '
output+=tmp
print output


# In[83]:


# generate a description for an image
#def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
max_length=34
in_text = 'startseq'
    # iterate over the whole length of the sequence
for i in range(max_length):
        # integer encode input sequence
    sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
    sequence = pad_sequences([sequence], maxlen=max_length)
    x_test_desc=pad_sequences([seq], maxlen=max_length)
    x_test_desc=torch.from_numpy(x_test_desc)
        #photo=Variable(photo)
    sequence=x_test_desc.type(torch.LongTensor)
    sequence=Variable(sequence)
    hoops=Xi[:1]
    #print photo
        # predict next word
    yhat = model(joops,sequence)
        # convert probability to integer
    yhat = torch.max(yhat,1,keepdim=False)
        # map integer to word
    #print yhat
    yhat=yhat[1]
    word = word_for_id(int(yhat.data), tokenizer)
        # stop if we cannot map the word
    if word is None:
        break
        # append as input for generating the next word
    in_text += ' ' + word
        # stop if we predict the end of the sequence
    if word == 'endseq':
        break
print in_text
    #return in_text


# In[58]:


generate_desc(model,tokenizer,joops,34)


# In[3]:


import torch
import torch.nn as nn


# In[16]:


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, 256)
        self.dropout = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(256, hidden_size, 1, batch_first=True)
        self.linear = nn.Linear(4096, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3= nn.Linear(256, 7579)
    
    def forward(self,features,captions):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        print "done"
        embeddings = self.dropout(embeddings)
        print "donee"
        lstmout, _ = self.lstm(embeddings)
        print "doneeee"
        features=self.dropout(features)
        features=self.linear(features)
        outputs=lstmout[:,-1,:]+features
        outputs=self.linear2(outputs)
        outputs=self.linear3(outputs)
        return outputs
model=DecoderRNN(256,7579)
model.load_state_dict(torch.load('model.pkl'))


# In[26]:


# generate a description for an image
#def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
max_length=34
joops=Xi[181:182]
in_text = 'startseq'
    # iterate over the whole length of the sequence
for i in range(max_length):
        # integer encode input sequence
    sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
    sequence = pad_sequences([sequence], maxlen=max_length)
    x_test_desc=sequence
    x_test_desc=torch.from_numpy(x_test_desc)
        #photo=Variable(photo)
    sequence=x_test_desc.type(torch.LongTensor)
    sequence=Variable(sequence)
    hoops=Xi[:1]
    #print photo
        # predict next word
    yhat = model(joops,sequence)
        # convert probability to integer
    yhat = torch.max(yhat,1,keepdim=False)
        # map integer to word
    #print yhat
    yhat=yhat[1]
    word = word_for_id(int(yhat.data), tokenizer)
        # stop if we cannot map the word
    if word is None:
        break
        # append as input for generating the next word
    in_text += ' ' + word
        # stop if we predict the end of the sequence
    if word == 'endseq':
        break
print in_text
    #return in_text

