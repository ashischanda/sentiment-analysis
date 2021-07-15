import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
import numpy as np
from collections import defaultdict

data = pd.read_csv('/data/train.csv')
print (data.head())

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

print (data_without_stopwords.head())

reviews = data_without_stopwords['clean_review']
reviews_trian_list = []
for i in range(len(reviews)):
  reviews_trian_list.append(reviews[i])
sentiment = data_without_stopwords['target']

y = np.array(list(map(lambda x: 1 if x==1 else 0, sentiment)))
X_train, X_validation,Y_train, Y_validation = train_test_split( reviews_trian_list, y, test_size=0.1 , random_state = 40)


word_index = dict()
index = 0


word_with_freq_more_than_one = 0
selected_word_index = dict()

for word_key , value in word.items():
    if value> 1:
        selected_word_index[ word_key ] = word_with_freq_more_than_one
        word_with_freq_more_than_one +=1


bow_matrix_train =[]
bow_matrix_validation =[]

# *********************************************** making bag of words matrix
for i in range (len( X_train) ):
    #word_list = [0] * ( len( selected_word_index ) + 1 )
    #word_list[ len( selected_word_index ) ] = len(  X_train[i].split() )  # keeping the length of words in a tweet as a feature
    
    word_list = [0] * len( selected_word_index )
    words = X_train[i].split()
    for w in words:
        if w in selected_word_index:
            index = selected_word_index[ w ]
            word_list [ index ] = 1
            
    bow_matrix_train.append( word_list)
# *********************************************** making bag of words matrix
for i in range (len( X_validation) ):
    #word_list = [0] * ( len( selected_word_index ) + 1 )
    #word_list[ len( selected_word_index ) ] = len(  X_validation[i].split() )  # keeping the length of words in a tweet as a feature
    
    
    word_list = [0] * len( selected_word_index )    
    words = X_validation[i].split()
    
    for w in words:
        if w in selected_word_index:
            index = selected_word_index[ w ]
            word_list [ index ] = 1
        
    bow_matrix_validation.append( word_list)
    
# *********************************************** 
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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


# Regularizer: l2 is used by default
logestic = LogisticRegression()
logestic.fit( bow_matrix_train, Y_train)
predictions = logestic.predict( bow_matrix_validation)

print ("")
print ("Model: Logistic regression")
arr = confusion_matrix( Y_validation, predictions) 
tn, fp, fn, tp = getArray( arr )

predict_proba = logestic.predict_proba( bow_matrix_validation)
z= roc_auc_score( Y_validation, predict_proba [ : ,1 ]) # It is a 2d array, take probability of class one
print ("AUC: " + str( round(z, 4)) )
print ("F1:  "+ str( getF1(tp, fn, fp, tn ) ) )
print ("Accuracy:" + str ( round( accuracy_score( Y_validation, predictions) , 4) ) )


# *****************************************************************************
# ****************************************************************************
print ("")
print ("Model: Decision Tree")
dTree = DecisionTreeClassifier( max_depth= 2)
dTree.fit( bow_matrix_train, Y_train)
predictions = dTree.predict( bow_matrix_validation)

arr = confusion_matrix( Y_validation, predictions) 
tn, fp, fn, tp = getArray( arr )


predict_proba = dTree.predict_proba( bow_matrix_validation)
z= roc_auc_score( Y_validation, predict_proba [ : ,1 ]) # It is a 2d array, take probability of class one
print ("AUC: " + str( round(z, 4)) )
print ("F1:  "+ str( getF1(tp, fn, fp, tn ) ) )
print ("Accuracy:" + str ( round( accuracy_score( Y_validation, predictions) , 4) ) ) 


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier( random_state=0, n_estimators = 10)  # n_estimators is The number of trees in the forest.
clf.fit( bow_matrix_train, Y_train)
predictions = clf.predict( bow_matrix_validation)
arr = confusion_matrix( Y_validation, predictions) 
tn, fp, fn, tp = getArray( arr )

print ("")
print ("Model: Random Forest")
predict_proba = clf.predict_proba( bow_matrix_validation)
z= roc_auc_score( Y_validation, predict_proba [ : ,1 ]) # It is a 2d array, take probability of class one
print ("AUC: " + str( round(z, 4)) )
print ("F1:  "+ str( getF1(tp, fn, fp, tn ) ) )
print ("Accuracy:" + str ( round( accuracy_score( Y_validation, predictions) , 4) ) ) 

