import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
import numpy as np
from collections import defaultdict

data = pd.read_csv('/home/ashis/ASHIS/Kaggle Competition/_twitter_disaster/data/train.csv')

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
reviews_trian_list = []
for i in range(len(reviews)):
  reviews_trian_list.append(reviews[i])
sentiment = data_without_stopwords['target']

y = np.array(list(map(lambda x: 1 if x==1 else 0, sentiment)))
X_train, X_validation,Y_train, Y_validation = train_test_split( reviews_trian_list, y, test_size=0.05 , random_state = 40)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import copy
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

hidden_size    = 768                    # For word2vec, we use 300
max_seq_length = 15   

y_train = pd.get_dummies(Y_train).values.tolist()  # Keeping each example in [1, 0]  format. The first value represent the presented class or label
y_test = pd.get_dummies(Y_validation).values.tolist()
  


# *****************************************************************************
from collections import defaultdict
GLOVE_EMB = '/glove.6B/glove.6B.300d.txt'

embeddings_ = dict()

f = open(GLOVE_EMB)
for line in f:
  values = line.split()
  word = values[0]
  coefs = np.asarray(values[1:], dtype='float32')
  embeddings_[word] = coefs
f.close()

print('Found %s word vectors.' %len(embeddings_))
   
word2vec_weight = []
word2index= dict()
token_dict= defaultdict( int )  # to make searching faster


# *****************************************************************************
dummy = [0] * hidden_size
word2vec_weight.append( dummy )    # index zero is for padding, adding zero in vector values
index = 0                          # 0 is used for non-tokens
# *****************************************************************************

for word, voca in embeddings_.items():  
    index+=1             
    word2index[ word ] = index
    token_dict[word] =   index
    word2vec_weight.append( embeddings_[word] )
    
word2vec_weight = np.array( word2vec_weight )
print ("\n total words : " + str(index))    
# *****************************************************************************



from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


#tokens_tensor = torch.tensor([indexed_tokens])
'''
tokens_tensor.shape
Out[3]: torch.Size([1, 14])
tensor([[  101,  2040,  2001,  3958, 27227,  1029,   102,  3958,   103,  2001,
          1037, 13997, 11510,   102]])
'''

# Load pre-trained model (weights)
my_bert_model = BertModel.from_pretrained('bert-base-uncased')
my_bert_model.eval()

#tokens_tensor = tokens_tensor.to('cuda')
my_bert_model.to('cuda')


# *********************  BERT classification model    *************************
# *********************  Simple linear classification *************************
# *****************************************************************************         
class BertForSequenceClassification(nn.Module):
  
    def __init__(self, num_labels=2 ):
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        hidden_dropout_prob_ashis = .05
        self.dropout = nn.Dropout( hidden_dropout_prob_ashis)
        self.classifier = nn.Linear( hidden_size, num_labels)
        
        nn.init.xavier_normal_(self.classifier.weight)
        
        print("\n **************** loading pretrained embeddings...")
      
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        
        _output=[]
        for single_batch_ids in range( len( input_ids) ):
           
            #print ("dkddkf")
            encoded_layers, bert_pooled_output = my_bert_model( input_ids[ single_batch_ids ] )
            
            last_hidden_states = encoded_layers[0]  # The last hidden-state is the first element of the output tuple
            #Sentence vector: Most people usually only take the hidden states of the [CLS] token of the last layer 
            cls_embeddings = last_hidden_states[0]
            
            #Case: 1    ACC: 0.60 (taking the embedding of  'cls')
            cls_embeddings_only_cls = cls_embeddings[0]
            cls_embeddings_only_cls = cls_embeddings_only_cls.flatten().tolist()  
            
            
            #Case: 2   ACC: 0.43 (taking sum of all word embedding)
            #taking sum of all the vectors
            #cls_embeddings = cls_embeddings.sum( dim =0 )
            #print (cls_embeddings.shape)
            #cls_embeddings = cls_embeddings.flatten().tolist()
            
            
            ### Case: 3 ACC: 0.62 (taking output embedding)
            #bert_pooled_output= bert_pooled_output[0]
            #bert_pooled_output = bert_pooled_output.flatten().tolist()  
            
            _output.append( cls_embeddings_only_cls )
            
        _output = np.array( _output )
        
        #print (_output )
        
        
#        print ( type(input_ids[0]))
#        #print ( input_ids[0] )
#        print ( input_ids[0] )
#        print ( input_ids[0].shape )
        #print ( input_ids.shape )           # batch by max_seq_length
        
        #loading word2vec embedding
        #pooled_output = self.embed( input_ids[0] )   # batch, max_seq_length, 768
        #pooled_output = pooled_output.sum(  1 )   # summing all 150 words
            # now, it is  (batch size, 768)
            
        
        
        #pooled_output = self.dropout(pooled_output) # batch_size, dim 
        #print ( _output.shape)
        
        #print ("hello")
        t = torch.from_numpy( _output ).float()
        logits = self.classifier(  t.to('cuda') )   # WE ALSO NEED TO MAKE THIS FLOAT TENSOR TO CUDA   # batch_size, num_labels = 2 
        #logits = self.classifier2(  logits )   # WE ALSO NEED TO MAKE THIS FLOAT TENSOR TO CUDA   # batch_size, num_labels = 2 
        
        #print (logits.shape)          
        
        return logits
    
'''
l = [ [2, 3, 1], [4, 5, 7], [0, 3, 8] ]
k = np.array( l)
t = torch.from_numpy( k).float()

torch.sum(outputs, dim=0) # size = [1, ncol]  #To sum over all rows (i.e. for each column):


a = torch.Tensor( [ [0,1,2], [3,4,5] ])
a.unsqueeze_(-1)
a = a.expand(2, 3, 1)
a.shape
Out[67]: torch.Size([2, 3, 1])


x = a.transpose(1, 2)
x.shape
Out[69]: torch.Size([2, 1, 3])


how do convert a torch.cuda.floatTensor type tensor to a torch.floatTensor. 
tem_tensor = tem_tensor.cpu()

how do convert a torch.floatTensor type tensor to a torch.cuda.floatTensor. 
tem_tensor = Variable( d ).cuda()

'''



class text_dataset(Dataset):
    def __init__(self,x_y_list, transform=None, bert_flag = True):       
        self.x_y_list = x_y_list
        self.transform = transform
        self.bert_flag = bert_flag
        
    def __getitem__(self,index):
        if self.bert_flag:
            # There are many slag words or random words for URL link 
            # Trying to avoid those STOP WORDS using glove dictionary
            # So, we are using same words for both models
            
            text = self.x_y_list[0][index]
            tokens = text.split()
            tokenized_review = [ t  for t in tokens if token_dict[t] !=0 ]
            
            tem_text = " ".join( tokenized_review )
            
            
            text = "[CLS] "+ tem_text +" [SEP]"           
            tokenized_text = tokenizer.tokenize(text)  
            if len(tokenized_text) > max_seq_length:
                tokenized_text = tokenized_text[:max_seq_length]                   
            ids_review  = tokenizer.convert_tokens_to_ids( tokenized_text )    

        else:
            text = self.x_y_list[0][index]
            tokens = text.split()
            tokenized_review = [ word2index[t]  for t in tokens if token_dict[t] !=0 ]
            
            if len(tokenized_review) > max_seq_length:
                tokenized_review = tokenized_review[:max_seq_length]
            ids_review  = tokenized_review 

        # *****************************************************************************
        # if length is smaller than max_seq_length, adding padding  
        # index zero is for padding
        padding = [0] * (max_seq_length - len(ids_review))            
        ids_review += padding            
        assert len(ids_review) == max_seq_length

        #print(ids_review)
        ids_review = torch.tensor( [ids_review] )
        
        sentiment = self.x_y_list[1][index] # color        
        list_of_labels = [torch.from_numpy(np.array(sentiment))]
        
        
        return ids_review, list_of_labels[0]
    
    def __len__(self):
        return len(self.x_y_list[0])

        
batch_size = 8
train_lists = [X_train, y_train]
test_lists =  [X_validation, y_test]

training_dataset = text_dataset(x_y_list = train_lists, bert_flag=  True )   #converting into bert index list
test_dataset =     text_dataset(x_y_list = test_lists , bert_flag = True )

dataloaders_dict = {'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
                    'val':torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0) }
dataset_sizes =    {'train':len(train_lists[0]), 'val':len(test_lists[0])}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

print(device)


num_labels = 2
embedding = True
model = BertForSequenceClassification(num_labels)


# *****************************************************************************
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    print('starting')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100
    best_acc = 0
    loss_updates_ten_times = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        val_pred =[]
        val_y =[]

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0    
            sentiment_corrects = 0
            
            
            # Iterate over data.
            interation = 0
            for inputs, sentiment in dataloaders_dict[phase]:
                interation+=1
                #print (interation)
                
                #inputs = inputs
                #print(len(inputs),type(inputs),inputs)
                #inputs = torch.from_numpy(np.array(inputs)).to(device) 
                inputs = inputs.to(device) 
                sentiment = sentiment.to(device)         
                optimizer.zero_grad()            # zero the parameter gradients

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #print(inputs)
                    outputs = model(inputs)
                    #print ("outp")
                    outputs = F.softmax(outputs,dim=1)
                   
                    
                    loss = criterion(outputs, torch.max(sentiment.float(), 1)[1] )
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

                
                sentiment_corrects += torch.sum(torch.max(outputs, 1)[1] == torch.max(sentiment, 1)[1])
                if phase =="val":
                    a = sentiment.flatten().tolist()   # For each example, there are two values for two classes
                    b = outputs.flatten().tolist()
                    #print (a)
               
                    for i in range (0, len(a)-1, 2 ):
                        ll = a[ i + 1]         # taking value of second class
                        val_y.append( ll )
                    for i in range (0, len(b)-1, 2 ):
                        ll = b[ i + 1]         # taking value of second class
                        val_pred.append( ll )
                
            epoch_loss = running_loss / dataset_sizes[phase]
            loss_updates_ten_times+=1

            
            sentiment_acc = sentiment_corrects.double() / dataset_sizes[phase]

            print('{} total loss: {:.4f} '.format(phase,epoch_loss ))
            print('{} sentiment_acc: {:.4f}'.format(phase, sentiment_acc))
            if phase == 'val' :
                pred_sentiment = np.array(list(map(lambda x : 1 if x > 0.5 else 0, val_pred)))
                arr = confusion_matrix( val_y, pred_sentiment) 
                tn, fp, fn, tp = getArray( arr )
                
                
                z= roc_auc_score( val_y, val_pred) # It is a 2d array, take probability of class one
                print ("Accuracy:" + str ( round( accuracy_score( val_y, pred_sentiment) , 4) ) )
                print ("AUC: " + str( round(z, 4)) )
                print ("F1:  "+ str( getF1(tp, fn, fp, tn ) ) )
                
            if phase == 'val' and epoch_loss < best_loss:
                print('saving with loss of {}'.format(epoch_loss), 'improved over previous {}'.format(best_loss))
                loss_updates_ten_times = 0
                best_loss = epoch_loss
                best_acc = sentiment_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'bert_model_test.pth')
            if loss_updates_ten_times ==10:  # There is no change of loss for last 10 times, then stop
                break


        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format( time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(float(best_loss)))
    print ("acc for that loss is "+ str( best_acc ) )

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model.to(device)

lrlast = .001
lrmain = .00001
optim1 = optim.Adam(
    [
   
        {"params":model.classifier.parameters(), "lr": lrlast},
       
   ])

#optim1 = optim.Adam(model.parameters(), lr=0.001)#,momentum=.9)
# Observe that all parameters are being optimized
optimizer_ft = optim1
criterion = nn.CrossEntropyLoss()

# Decay Learning Rate by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

# *****************************************************************************
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
# *****************************************************************************
    
model_ft1 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=  100)
