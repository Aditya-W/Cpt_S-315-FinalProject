
import pandas as pd
import string
import nltk

data = pd.read_csv('spam.txt', sep = '\t', header=None, names=["label", "sms"])
data.head()

nltk.download('stopwords')
nltk.download('punkt')

# variable containing stop words in english. Used to filter our dataset
stopwords = nltk.corpus.stopwords.words('english')
# variable representing punctuation in english language. Will be used later 
punctuation = string.punctuation


# this function is used for pre-processing words
def processWords(sms):
    # removing punctuation from the input
    rmPunctuation = "".join([word.lower() for word in sms if word not in punctuation])
    # tokenizing words
    tokenizedWords = nltk.tokenize.word_tokenize(rmPunctuation)
    # removing stop words from the input
    rmStopWords = [word for word in tokenizedWords if word not in stopwords]
    return rmStopWords

# adding a column to our data with our processed messages
data['processed'] = data['sms'].apply(lambda x: processWords(x))

# this function is used to categorize words
def categorize():
    spam = [] # spam words
    ham = [] # ham words
    #handling messages associated with spam
    for sms in data['processed'][data['label'] == 'spam']:
        for word in sms:
            spam.append(word)
    #handling messages associated with ham
    for sms in data['processed'][data['label'] == 'ham']:
        for word in sms:
            ham.append(word)
    return spam, ham

spam, ham = categorize()

# this function does the job of predicitng whether a message is a spam or a ham
def predict(sms):
    sCounter = 0
    hCounter = 0
    #count the occurances of each word in the sms string
    for word in sms:
        sCounter += spam.count(word)
        hCounter += ham.count(word)
    print('***RESULTS***')
    #if the message is ham
    if hCounter > sCounter:
        accuracy = round((hCounter / (hCounter + sCounter) * 100))
        print('messege is not spam, with {}% certainty'.format(accuracy))
    #if the message is equally spam and ham
    elif hCounter == sCounter:
        print('message could be spam')
    #if the message is spam
    else:
        accuracy = round((sCounter / (hCounter + sCounter)* 100))
        print('message is spam, with {}% certainty'.format(accuracy))




userInput = input("Please input a spam or a ham message to check this function\n")
#pre-processing the input before prediction
processed_input = processWords(userInput)

# outputting final prediction
predict(processed_input)







