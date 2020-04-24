import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from sklearn.model_selection import train_test_split

dataM = pd.read_csv("Files/Gender_Recog/Dataset/Female-Names.csv")
dataF = pd.read_csv("Files/Gender_Recog/Dataset//Male-Names.csv")

df = pd.concat([dataM,dataF])

#df.head()

def should_keep(word):
  if(len(word)) > 19:
    return False
  char_set = [' ', '.', '1', '0', '3', '2', '5', '4', '7', '6', '9', '8', 'END', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i', 'h', 'k', 'j', 'm', 'l', 'o', 'n', 'q', 'p', 's', 'r', 'u', 't', 'w', 'v', 'y', 'x', 'z']  
  for ch in word:
    if ch not in set(char_set):
      return False
  return True
  

def clean(word):
  name = str(word)
  name = name.lower()
  if should_keep(name):
    return name
  else:
    return None 
df.name = df.name.apply(lambda word : clean(word))
  

#df.head(10)

df = df.dropna()

char_set = [' ', '.', '1', '0', '3', '2', '5', '4', '7', '6', '9', '8', 'END', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i', 'h', 'k', 'j', 'm', 'l', 'o', 'n', 'q', 'p', 's', 'r', 'u', 't', 'w', 'v', 'y', 'x', 'z']
char2idx = {}
index = 0
for ch in char_set:
  char2idx[ch] = index
  index+=1

df = df.drop('race',axis=1)
vector_length = 39
max_word_len = 20
words = []
labels= []
for name,gender in df.itertuples(index=False):
  one_hots_word = []
  
  for ch in name:
    vec = np.zeros(vector_length)
    vec[char2idx[ch]] = 1
    one_hots_word.append(vec)
  for _ in range(max_word_len - len(name)):
    vec = np.zeros(vector_length)
    vec[char2idx['END']] = 1
    one_hots_word.append(vec)
  one_hots_word = np.array(one_hots_word)
  words.append(one_hots_word)
  labels.append(gender)


words = np.array(words)
#words.shape

#len(labels)

labels = np.array(labels)
one = OneHotEncoder()
labels_one_hot = one.fit_transform(labels.reshape(-1 , 1)).todense()

model = Sequential()
model.add(LSTM(128,input_shape=(20,39),return_sequences=True))
model.add(LSTM(120))
model.add(Dropout(.3))
model.add(Dense(2,activation='softmax'))
#model.summary()

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

X_train, X_test, y_train, y_test = train_test_split(words,labels_one_hot , test_size=0.33, random_state=42)

#model.fit(X_train,y_train,epochs=5)
#model.evaluate(X_test,y_test)
def word2vec(urname):
  char_set = [' ', '.', '1', '0', '3', '2', '5', '4', '7', '6', '9', '8', 'END', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
              'h', 'k', 'j', 'm', 'l', 'o', 'n', 'q', 'p', 's', 'r', 'u', 't', 'w', 'v', 'y', 'x', 'z']
  char2idx = {}
  index = 0
  for ch in char_set:
    char2idx[ch] = index
    index += 1
  vector_length = 39
  max_word_len = 20
  words = []
  one_hots_word = []

  for ch in urname:
    vec = np.zeros(vector_length)
    vec[char2idx[ch]] = 1
    one_hots_word.append(vec)
  for _ in range(max_word_len - len(urname)):
    vec = np.zeros(vector_length)
    vec[char2idx['END']] = 1
    one_hots_word.append(vec)
  one_hots_word = np.array(one_hots_word)

  return one_hots_word



def analyzer(sent):
  name = sent.lower()
  cleaned_mat = word2vec(name)
  # cleaned_mat.shape
  cleaned_mat = cleaned_mat.reshape(1, 20, 39)
  model.predict(cleaned_mat)
  npp = np.argmax(model.predict(cleaned_mat))
  if npp == 0:
    npp = "Female"
  else:
    npp = "Male"
  return(npp)






