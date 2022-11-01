import nltk
#NLTK-Natural Language Tool Kit
#pip install nltk
text='Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract or extrapolate knowledge and insights from noisy, structured and unstructured data,[1][2] and apply knowledge from data across a broad range of application domains. Data science is related to data mining, machine learning and big data.[3]'


#inside NLTK we have sentence tokenizer, it will break entire paragraph into sentences based on the fullstop found.
print(nltk.sent_tokenize(text)) #gives the list of sentences
print(nltk.sent_tokenize(text)[1])

#Similar to sencentence tokenizer we have word tokenizer as well inside nltk. Which will break down each word and
#gives the list of words output
print(nltk.word_tokenize(text))
print(nltk.word_tokenize(text)[2])

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
#Nltk is a rule based library where you can find the gramitical parameters such as verb, noun, adjective,synanims...etc
#POS=parts of speach
text1="We are trying to print the NLTK tokenizer POS details"
print(nltk.pos_tag([text1,])) #gives the POS tokenize of each character inside the text
#to get POS of words in the text
print(nltk.pos_tag(nltk.word_tokenize(text1)))

#I,we,me,the, comma, full stop, am, these words or information are not going to contribute for the finding the
#relationship while building the model. we need to remove these using python code or using nltk corpus stopwords.
nltk.download('stopwords')
from nltk.corpus import stopwords
print(stopwords.words('english')) # this will list the stop words of english language.
stop_words=stopwords.words('english')
#we need to build a logic to compare our word list with this stopwords list and remove them from our list
#you need to different different corpus stopword library for different different languages

import string
print(string.punctuation) # to get the list of punctuations which are also need to be removed
stop_punch=string.punctuation
l=[]
for i in nltk.word_tokenize(text):
    if i not in stop_punch:
        if i not in stop_words:
            l.append(i)

print('text without stop words and punctuation   ', l)

print(nltk.pos_tag(l))

#some words as run running...etc if present in our text, then during model building the system will treate them as
#two different words which will impact on finding the relation ship for our sentimental analyisis of our data
#so we need to perform limitization of stemming to remove such things.

from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer, SnowballStemmer

Porter_obj=PorterStemmer()
lancast_obj=LancasterStemmer()
snow_obj=SnowballStemmer('english')
print(Porter_obj.stem('running')) #It will shop of ing, ies, es etc from particular world, this will help you to
        #convert your text with the base word for example instead of running run. Babies-babi...etc
print(Porter_obj.stem('Babies'))
print(Porter_obj.stem('Babi'))
print(lancast_obj.stem('Running')) # even lancaster and snow ball does the same kind of chopping but with little bit
print(snow_obj.stem('Running'))# difference.

#now let us see with Lemmatizer. The Lemmatizer does not chop the ending letters but it will try to convert the
#given word to its base word, for example running to run, runs to run...etc. These libraries are case sensitives
# you have to give in lower letters.
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
lemma_obj=WordNetLemmatizer()
print(lemma_obj.lemmatize('Running'))
print(lemma_obj.lemmatize('Runs'))
print(lemma_obj.lemmatize('Run'))

print(lemma_obj.lemmatize('went'))
print(lemma_obj.lemmatize('gone'))
print(lemma_obj.lemmatize('go'))

print(lemma_obj.lemmatize('went',pos='v')) #Now based on the part of speach it is converting it in to base word here.

#Problem statment : Example: We have text containing the multiple reviews of iNeuron, some of them are possitive
#reviews and some are negative reviews. Now we need to build the model such that if any review text is fed to it
#it should say possitive or negetaive review.
#The approach should be as below
# The data given is having unstructured data i.e. some comments are of single line, others are in mulitple line
#Our first objective should be convert this data into structured numerical format.
#How to convert this infor into strucutered numerical fomat? TFIDF- Term frequency Inverse document frequency
#TFIDF- Steps
#Convert the sentences in to word tokenizers
#Then apply set on it, you will get unique words to get the column names for building the TFIDF table
#Create the number of rows as number of documents or number of messages. Here we have 3 messages so three rows
#TF= Term frequency= Term i frequency in document j/ Total word in document j
#IDF= Inverse docment frequency= log(Total document/document with term j)


















