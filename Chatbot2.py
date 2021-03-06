import pickle
import numpy as np

#get train and test data
#################################################
#opens files containing tuples of stories
#for use in training data for DL
with open('train_qa.txt','rb') as f:
    train_data = pickle.load(f)

#for use in test data for DL
with open('test_qa.txt','rb') as f:
    test_data = pickle.load(f)

#combines train and test data
all_data = test_data + train_data

#create vocabulary
###################################################

#creates empty set for vocab to use
vocab = set()

#eliminates the tuple separation within test data and finds disctinct elements
for story,question,answer in all_data:
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))
#test and visualization of vocab size
vocab.add('no')
vocab.add('yes')

#adds one to the length for a place holder for when we use keiras sequences
vocab_len = len(vocab)+1


all_story_lens = [len(data[0]) for data in all_data]

#creates a list of the complete story vocab and designates the sequential length
max_story_len = max(all_story_lens)

#same for the longest question.
max_question_len = max([len(data[1]) for data in all_data])

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

#creat an instance of tokenizer
tokenizer = Tokenizer(filters=[])
tokenizer.fit_on_texts(vocab)

#empty lists and seperation of story question and answer
train_story_text = []
train_question_text = []
train_answers = []

for story,question,answer in train_data:
    train_story_text.append(story)
    train_question_text.append(question)
    train_answers.append(answer)

train_story_seq = tokenizer.texts_to_sequences(train_story_text)

#function to vectorize the data
def vectorize_stories(data,word_index= tokenizer.word_index , max_story_len = max_story_len, max_question_len = max_question_len):
    #Stories or inputs are X
    X = []

    #Question inputs = Xq
    Xq = []

    # Question answer or outputs yes or no = Y
    Y = []

    for story,query,answer in data:

        #creates a list for each story
        x = [word_index[word.lower()]for word in story]

        #same thing for questions
        xq = [word_index[word.lower()]for word in query]

        #since the answer is just yes or no 
        y = np.zeros(len(word_index)+1)

        y[word_index[answer]] = 1

        #add to those empty lists
        X.append(x)
        Xq.append(xq)
        Y.append(y)

    #return the padding between sentences
    return (pad_sequences(X,maxlen=max_story_len),pad_sequences(Xq,maxlen= max_question_len),np.array(Y))

inputs_train, queries_train, answers_train = vectorize_stories(train_data)

inputs_test , queries_test, answers_test = vectorize_stories(test_data)


#ENCODING
#####################################################################################################
from keras.models import Sequential,Model
from keras.layers.embeddings import Embedding
from keras.layers import Input,Activation,Dense,Permute,Dropout,add,dot,concatenate,LSTM

# Placeholder that takes in the inputs in a shape(story length by batch size)
input_sequence = Input((max_question_len,))
question = Input((max_question_len,))

#create input encoders
vocab_size = len(vocab) + 1

#Encoder M
input_encoder_m = Sequential()
#Adds embedding layer with input dimension of the vocab size and the output dimension of 64
input_encoder_m.add(Embedding(input_dim= vocab_size,output_dim = 64))
#turns off percentage of nuerons randomly while training (can be experimented with)
input_encoder_m.add(Dropout(0.3))

#outputs of (samples,story_maxlen,embedding_dim)

#ENCODER C FOr question
input_encoder_c = Sequential()
#Adds embedding layer with input dimension of the vocab size and the output dimension of the question length
input_encoder_c.add(Embedding(input_dim= vocab_size,output_dim = max_question_len))
#turns off percentage of nuerons randomly while training (can be experimented with)
input_encoder_c.add(Dropout(0.3))

#outputs of (samples,story_maxlen,max_question_len)

#question Encoder
question_encoder = Sequential()
#Adds embedding layer with input dimension of the vocab size and the output dimension of encoder M and input length = encoder c
question_encoder.add(Embedding(input_dim= vocab_size,output_dim = 64,input_length= max_question_len))
#turns off percentage of nuerons randomly while training (can be experimented with)
question_encoder.add(Dropout(0.3))

#result of passing it through encoder
input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)

match = dot([input_encoded_m,question_encoded],axes=(2,2))
match = Activation('softmax')(match)
response = add([match,input_encoded_c])
response = Permute((2,1))(response)

answer = concatenate([response,question_encoded])

answer = LSTM(32)(answer)
answer = Dropout(0.5)(answer)
answer = Dense(vocab_size)(answer)

#ouptuts shape of samples by vocab size and outputs yes no

#ouput probability distribution over yes and no
answer = Activation('softmax')(answer)
model = Model([input_sequence,question],answer)
model.compile(optimizer ='rmsprop',loss= 'categorical_crossentropy',metrics = ['accuracy'])

#FITTING AND TRAINING MODEL
history = model.fit([inputs_train,queries_train],answers_train,batch_size=32,epochs=3,validation_data=([inputs_test,queries_test],answers_test))

#training model
model.load_weights('chatbot_10.5')
pred_results= model.predict(([inputs_test,queries_test]))

#the predicted answers to the questions
val_max = np.argmax(pred_results[0])

#indexed items
for key,val in tokenizer.word_index.items():
    if val == val_max:
        k = key

#handles inputs for questions
#Only handles words and questions within the vocabulary

my_story = "John left the kitchen . Sandra dropped the football in the garden . "

my_question  = "is the football in the garden ?"
mydata = [(my_story.split(),my_question.split(),'yes')]

#everything vectorized seperately
my_story, my_ques, my_ans = vectorize_stories(mydata)

#Final output(myans) of my inputs (mystory,myquestion,)

pred_results = model.predict(([my_story,my_ques]))
val_max = np.argmax(pred_results[0])

for key,val in tokenizer.word_index.items():
    if val == val_max:
        k = key