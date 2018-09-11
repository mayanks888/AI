
# coding: utf-8

# # Dense Sentiment Classifier

# In this notebook, we build a dense neural net to classify IMDB movie reviews by their sentiment.

# #### Load dependencies

# In[1]:


import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Embedding # new!
from keras.callbacks import ModelCheckpoint # new! 
import os # new! 
from sklearn.metrics import roc_auc_score, roc_curve # new!
import pandas as pd
import matplotlib.pyplot as plt # new!
# get_ipython().run_line_magic('matplotlib', 'inline')


# #### Set hyperparameters

# In[2]:


# output directory name:
output_dir = 'model_output/dense'

# training:
epochs = 4
batch_size = 128

# vector-space embedding: 
n_dim = 64
n_unique_words = 5000 # as per Maas et al. (2011); may not be optimal
n_words_to_skip = 50 # ditto
max_review_length = 100
pad_type = trunc_type = 'pre'

# neural network architecture: 
n_dense = 64
dropout = 0.5


# #### Load data

# For a given data set: 
# 
# * the Keras text utilities [here](https://keras.io/preprocessing/text/) quickly preprocess natural language and convert it into an index
# * the `keras.preprocessing.text.Tokenizer` class may do everything you need in one line:
#     * tokenize into words or characters
#     * `num_words`: maximum unique tokens
#     * filter out punctuation
#     * lower case
#     * convert words to an integer index

# In[3]:


(x_train, y_train), (x_valid, y_valid) = imdb.load_data(num_words=n_unique_words, skip_top=n_words_to_skip) 


# In[4]:


x_train[0:6] # 0 reserved for padding; 1 would be starting character; 2 is unknown; 3 is most common word, etc.


# In[5]:


for x in x_train[0:6]:
    print(len(x))


# In[6]:


y_train[0:6]


# In[7]:


len(x_train), len(x_valid)


# #### Restoring words from index

# In[8]:


word_index = keras.datasets.imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["PAD"] = 0
word_index["START"] = 1
word_index["UNK"] = 2


# In[9]:


word_index


# In[10]:


index_word = {v:k for k,v in word_index.items()}


# In[11]:


x_train[0]


# In[12]:


' '.join(index_word[id] for id in x_train[0])


# In[13]:


(all_x_train,_),(all_x_valid,_) = imdb.load_data() 


# In[14]:


' '.join(index_word[id] for id in all_x_train[0])


# #### Preprocess data

# In[15]:


x_train = pad_sequences(x_train, maxlen=max_review_length, padding=pad_type, truncating=trunc_type, value=0)
x_valid = pad_sequences(x_valid, maxlen=max_review_length, padding=pad_type, truncating=trunc_type, value=0)


# In[16]:


x_train[0:6]


# In[17]:


for x in x_train[0:6]:
    print(len(x))


# In[18]:


' '.join(index_word[id] for id in x_train[0])


# In[19]:


' '.join(index_word[id] for id in x_train[5])


# #### Design neural network architecture

# In[20]:


model = Sequential()
model.add(Embedding(n_unique_words, n_dim, input_length=max_review_length))
model.add(Flatten())
model.add(Dense(n_dense, activation='relu'))
model.add(Dropout(dropout))
# model.add(Dense(n_dense, activation='relu'))
# model.add(Dropout(dropout))
model.add(Dense(1, activation='sigmoid')) # mathematically equivalent to softmax with two classes


# In[21]:


model.summary() # so many parameters!


# In[22]:


# embedding layer dimensions and parameters: 
n_dim, n_unique_words, n_dim*n_unique_words


# In[23]:


# ...flatten:
max_review_length, n_dim, n_dim*max_review_length


# In[24]:


# ...dense:
n_dense, n_dim*max_review_length*n_dense + n_dense # weights + biases


# In[25]:


# ...and output:
n_dense + 1 


# #### Configure model

# In[26]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[27]:


modelcheckpoint = ModelCheckpoint(filepath=output_dir+"/weights.{epoch:02d}.hdf5")


# In[28]:


if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# #### Train!

# In[29]:


# 84.7% validation accuracy in epoch 2
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_valid, y_valid), callbacks=[modelcheckpoint])


# #### Evaluate

# In[30]:


model.load_weights(output_dir+"/weights.01.hdf5") # zero-indexed


# In[31]:


y_hat = model.predict_proba(x_valid)


# In[32]:


len(y_hat)


# In[33]:


y_hat[0]


# In[34]:


plt.hist(y_hat)
_ = plt.axvline(x=0.5, color='orange')


# In[35]:


pct_auc = roc_auc_score(y_valid, y_hat)*100.0


# In[36]:


"{:0.2f}".format(pct_auc)


# In[37]:


float_y_hat = []
for y in y_hat:
    float_y_hat.append(y[0])


# In[38]:


ydf = pd.DataFrame(list(zip(float_y_hat, y_valid)), columns=['y_hat', 'y'])


# In[39]:


ydf.head(10)


# In[40]:


' '.join(index_word[id] for id in all_x_valid[0])


# In[41]:


' '.join(index_word[id] for id in all_x_valid[6]) 


# In[42]:


ydf[(ydf.y == 0) & (ydf.y_hat > 0.9)].head(10)


# In[43]:


' '.join(index_word[id] for id in all_x_valid[489]) 


# In[44]:


ydf[(ydf.y == 1) & (ydf.y_hat < 0.1)].head(10)


# In[45]:


' '.join(index_word[id] for id in all_x_valid[927]) 

