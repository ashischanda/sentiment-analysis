'''
You can use the same LSTM model for Skip-gram, FastText and BERT model
You just need to update the pre-trained embedding file (i.e. GLOVE_EMB)
'''

from keras.datasets import imdb
import pandas as pd
import numpy as np
from keras.layers import LSTM, Activation, Dropout, Dense, Input,Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.models import Model
import string
import re
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.sequence import pad_sequences
import keras
from sklearn.model_selection import train_test_split

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)


downloaded = drive.CreateFile({'id':"1qlTo_jR46mYydQaS70TgS3Ghx00009bk"})   # replace the ID with your fiie ID from Google drive
downloaded.GetContentFile('train.csv') 


data = pd.read_csv('data/train.csv')

data['text'] = data['text'].str.lower()
stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", 
             "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during",
             "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", 
             "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
             "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or",
             "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", 
             "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
             "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up",
             "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's",
             "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've",
             "your", "yours", "yourself", "yourselves" ]

def remove_stopwords(data):
  data['review without stopwords'] = data['text'].apply(lambda x : ' '.join([word for word in x.split() if word not in (stopwords)]))
  return data

def remove_tags(string):
    result = re.sub('<.*?>','',string)
    return result

data_without_stopwords = remove_stopwords(data)
data_without_stopwords['clean_review']= data_without_stopwords['review without stopwords'].apply(lambda cw : remove_tags(cw))
data_without_stopwords['clean_review'] = data_without_stopwords['clean_review'].str.replace('[{}]'.format(string.punctuation), ' ')

reviews = data_without_stopwords['clean_review']
reviews_list = []
for i in range(len(reviews)):
  reviews_list.append(reviews[i])
sentiment = data_without_stopwords['target']

y = np.array(list(map(lambda x: 1 if x==1 else 0, sentiment)))
X_train, X_test,Y_train, Y_test = train_test_split(reviews_list, y, test_size=0.1, random_state = 40)

tokenizer = Tokenizer(num_words=6000)
tokenizer.fit_on_texts(X_train)
words_to_index = tokenizer.word_index
print (len(words_to_index))

def read_glove_vector(glove_vec):
  with open(glove_vec, 'r', encoding='UTF-8') as f:
    words = set()
    word_to_vec_map = {}
    for line in f:
      w_line = line.split()
      curr_word = w_line[0]
      word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)



  return word_to_vec_map



# *************   Loading pre-trained word2vec on Wiki_100MB data  ************
# *****************************************************************************
from collections import defaultdict
# https://drive.google.com/file/d/1O8nhcOu3PNtWpdr49zNi6wNRT1XaoUDB/view?usp=sharing
GLOVE_EMB = '/glove.6B/glove.6B.300d.txt'
downloaded = drive.CreateFile({'id':"1O8nhcOu3PNtWpdr49zNi6wNRT00aoUDB"})
downloaded.GetContentFile('glove300D.txt') 
GLOVE_EMB ='glove300D.txt'

word_to_vec_map = read_glove_vector(GLOVE_EMB)

maxLen = 15

vocab_len = len(words_to_index)
embed_vector_len = word_to_vec_map['moon'].shape[0]

emb_matrix = np.zeros((vocab_len, embed_vector_len))

for word, index in words_to_index.items():
  embedding_vector = word_to_vec_map.get(word)
  if embedding_vector is not None:
    emb_matrix[index, :] = embedding_vector

embedding_layer = Embedding(input_dim=vocab_len, output_dim=embed_vector_len, input_length=maxLen, weights = [emb_matrix], trainable=False)

from keras.layers import Bidirectional
def sentiment_analysis(input_shape):

  X_indices = Input(input_shape)

  embeddings = embedding_layer(X_indices)

  X = Bidirectional(LSTM( 15, return_sequences=True, recurrent_dropout=0.1))(embeddings)  
  X = Dropout(0.3)(X)
  X = LSTM(15)(X)
  X = Dropout(0.3)(X)

  X = Dense(1, activation='sigmoid')(X)

  model = Model(inputs=X_indices, outputs=X)

  return model

model = sentiment_analysis((maxLen,))
print (model.summary())

X_train_indices = tokenizer.texts_to_sequences(X_train)

X_train_indices = pad_sequences(X_train_indices, maxlen=maxLen, padding='post')
X_train_indices.shape

X_test_indices = tokenizer.texts_to_sequences(X_test)
X_test_indices = pad_sequences(X_test_indices, maxlen=maxLen, padding='post')

adam = keras.optimizers.Adam(learning_rate = 0.0001)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

trained_model = model.fit(X_train_indices, Y_train, validation_data=(X_test_indices, Y_test), batch_size=256, epochs=100)
pred = model.predict( X_test_indices )


# **********************************************************************************
# ********************* Reading validation data to get prediction result *****************
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


def get_precision(  tp, fn, fp, tn ):
    return round( tp/(tp+fp) ,4)

def get_recall(  tp, fn, fp, tn ):
    return round( tp/(tp+fn) , 4) 

def getF1(  tp, fn, fp, tn ):
    precision= get_precision(tp, fn, fp, tn)
    recall =get_recall(tp, fn, fp, tn)
    
    F1 = 2 * (precision * recall) / (precision + recall)
    return round(F1, 4)
    

def getArray(arr):
    tn = arr[0][0]
    fp = arr[0][1]
    fn = arr[1][0]
    tp = arr[1][1]

    return tn, fp, fn, tp

pred_sentiment = np.array(list(map(lambda x : 1 if x > 0.5 else 0, pred)))
arr = confusion_matrix( Y_test, pred_sentiment) 
tn, fp, fn, tp = getArray( arr )
                     
z= roc_auc_score( Y_test, pred) # It is a 2d array, take probability of class one
print ("Accuracy:" + str ( round( accuracy_score( Y_test, pred_sentiment) , 4) ) )
print ("AUC: " + str( round(z, 4)) )
print ("F1:  "+ str( getF1(tp, fn, fp, tn ) ) )

# **********************************************************************************
# ********************* Reading test file to get prediction result *****************
test_file = "https://drive.google.com/file/d/1VdQMlaV-2fqkP1-NoADC_bzUAdaY6/view?usp=sharing"
test_file ='test.csv'
downloaded = drive.CreateFile({'id':"1VdQMlaV-2fqkP1-NoADC_bzUAdaY6"}) 
downloaded.GetContentFile(test_file) 


tdata = pd.read_csv(test_file)
print (tdata.head())
tdata['text'] = tdata['text'].str.lower()

tdata_without_stopwords = remove_stopwords(tdata)
tdata_without_stopwords['clean_review']= tdata_without_stopwords['review without stopwords'].apply(lambda cw : remove_tags(cw))
tdata_without_stopwords['clean_review'] = tdata_without_stopwords['clean_review'].str.replace('[{}]'.format(string.punctuation), ' ')

print (tdata_without_stopwords.head())

reviews = tdata_without_stopwords['clean_review']
reviews_list = []
for i in range(len(reviews)):
  reviews_list.append(reviews[i])


X_maintest_indices = tokenizer.texts_to_sequences( reviews_list )
X_maintest_indices = pad_sequences(X_maintest_indices, maxlen=maxLen, padding='post')
print (len( X_maintest_indices))

preds = model.predict(X_maintest_indices)

pred_sentiment = np.array(list(map(lambda x : 1 if x > 0.5 else 0, preds)))
#print (pred_sentiment)


# **********************************************************************************
# ********************* Writing test prediction result in a file   *****************
from google.colab import drive
drive.mount('/content/gdrive')

writer = open('/content/gdrive/My Drive/file.txt', "w")
for value in pred_sentiment:
  writer.write( str(value))
  writer.write("\n")
writer.close()  
print ("Finished!")
