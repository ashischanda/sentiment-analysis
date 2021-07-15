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
X_train, X_validation,Y_train, Y_validation = train_test_split( reviews_trian_list, y, test_size=0.05 , random_state = 40)

print ("*" * 20)
print (len(Y_train))
print (len(Y_validation))

# *****************************************************************************
# ********************  Calculating avg. word length in tweets ****************

ss = 0 
m = 0
n = 99999999999
li = []
word = defaultdict( int )
positive_comment_words = ""
negative_comment_words = ""
count_positive_samples = 0

word_index = dict()
index = 0
length_list = []
length_list_p = []
length_list_n = []

for i in range (len( reviews_trian_list) ):
  if sentiment[i] ==1:
      positive_comment_words +=  reviews_trian_list[i] 
      count_positive_samples +=1
      length_list_p.append( len( reviews_trian_list[i].split() )  )
  else:
      negative_comment_words +=  reviews_trian_list[i] 
      length_list_n.append( len( reviews_trian_list[i].split() )  )
      
      
  words = reviews_trian_list[i].split()
  for w in words:
      word[w] +=1
      if w not in word_index:
          word_index[ w ] = index
          index+=1
  length_list.append( len( reviews_trian_list[i].split() ) )    
  ss += len( reviews_trian_list[i].split() )
  if m < len( reviews_trian_list[i].split() ):
    m = len( reviews_trian_list[i].split() )
  if n > len( reviews_trian_list[i].split() ):
    n = len( reviews_trian_list[i].split() )
  li.append( len( reviews_trian_list[i].split())  )

print ("Average words per line:  ")
print (ss/len(reviews_trian_list)) # 14.9
print (m)               # 31 
print (n)
li = sorted( li)
print ("Median length: "+ str( li[ int(len(li)/2) ] ) )
print ("Total unique words: "+ str( len(word) ) )

word_with_freq_more_than_one = 0
for key , value in word.items():
    if value> 1:
        word_with_freq_more_than_one +=1

print ("Total unique words with frequency >1: "+ str( word_with_freq_more_than_one ) )


print ("Total positive samples: "+ str( count_positive_samples ) )

print ("*" * 20)
print ("Baseline accuracy: " +  str( round( max( [count_positive_samples, len(reviews_trian_list) - count_positive_samples] )/len(reviews_trian_list), 2    )) )

# *****************************************************************************
# ******************  showing histogram of word length in tweets **************

# *******************************************************
import math
import matplotlib.pyplot as plt 

def drawHistogram(x, x2, xlabel="", ylabel="" , title="graph"):
    labels=["positive","negative"]
    fig= plt.figure()
    plt.hist([x, x2] , bins= 35)
    #plt.hist(x2 ,bins= 35)
    
    plt.ylabel( ylabel)
    plt.xlabel( xlabel)
    plt.title( title )
    plt.legend(labels)
    #plt.legend('avvvv')
    fig.savefig( title+".png" ,   dpi=fig.dpi )  # dpi is for setting the image in deep pixel?
    
    plt.show()
# *******************************************************

drawHistogram( length_list_p, length_list_n , "Twitter length", "Frequency or count (tweets)", "")

'''

# *****************************************************************************
# ******************  showing word cloud of for tweets ************************

# importing all necessery modules
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate( negative_comment_words )
  
# plot the WordCloud image                       
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
  
plt.show()
'''
