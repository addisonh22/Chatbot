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

print(sum(answers_test))