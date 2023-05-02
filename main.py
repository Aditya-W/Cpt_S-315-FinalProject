#!/usr/bin/env python
# coding: utf-8

# ## Step 1: Loading the Dataset

# In[12]:


import pandas as pd

data = pd.read_csv('spam.txt', sep = '\t', header=None, names=["label", "sms"])
data.head()


# ## Step 2: Pre-Processing

# In[3]:


import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')

stopwords = nltk.corpus.stopwords.words('english')
punctuation = string.punctuation

print(stopwords[:5])
print(punctuation)


# In[14]:


def pre_process(sms):
    remove_punct = "".join([word.lower() for word in sms if word not in punctuation])
    tokenize = nltk.tokenize.word_tokenize(remove_punct)
    remove_stopwords = [word for word in tokenize if word not in stopwords]
    return remove_stopwords

#adding a column to our data with our processed messages
data['processed'] = data['sms'].apply(lambda x: pre_process(x))

print(data['processed'].head())


# ## Step 3: Categorizing and Counting Tokens

# In[5]:


def categorize_words():
    spam_words = []
    ham_words = []
    #handling messages associated with spam
    for sms in data['processed'][data['label'] == 'spam']:
        for word in sms:
            spam_words.append(word)
    #handling messages associated with ham
    for sms in data['processed'][data['label'] == 'ham']:
        for word in sms:
            ham_words.append(word)
    return spam_words, ham_words

spam_words, ham_words = categorize_words()

print(spam_words[:5])
print(ham_words[:5])


# ## Step 4: Predict Function

# In[6]:


def predict(sms):
    spam_counter = 0
    ham_counter = 0
    #count the occurances of each word in the sms string
    for word in sms:
        spam_counter += spam_words.count(word)
        ham_counter += ham_words.count(word)
    print('***RESULTS***')
    #if the message is ham
    if ham_counter > spam_counter:
        accuracy = round((ham_counter / (ham_counter + spam_counter) * 100))
        print('messege is not spam, with {}% certainty'.format(accuracy))
    #if the message is equally spam and ham
    elif ham_counter == spam_counter:
        print('message could be spam')
    #if the message is spam
    else:
        accuracy = round((spam_counter / (ham_counter + spam_counter)* 100))
        print('message is spam, with {}% certainty'.format(accuracy))


# ## Step 5: Collecting User Input & Results

# In[11]:


user_input = input("Please type a spam or ham message to check if our function predicts accurately\n")
#pre-processing the input before prediction
processed_input = pre_process(user_input)

predict(processed_input)


# In[ ]:




