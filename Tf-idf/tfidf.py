import nltk,sys
import numpy as np,pandas as pd
import matplotlib.pyplot as plt 
import nltk.data
import csv
from nltk.stem.porter import *
pd.options.mode.chained_assignment = None

data = pd.read_csv("mbti_1.csv") 
shortdata=data.iloc[:,-1]
shortdata=shortdata.head()
shortdata1=data.iloc[0:5,0]
print('-----Data-------')
print(shortdata)

#removing stopwords 
from nltk.corpus import stopwords
stop=stopwords.words("english"),'I'
print('------Removing stopwords------')
shortdata=shortdata.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#shortdata=shortdata.apply(lambda x: ' '.join([word for word in x.split() if word!='i' or word!='I']))
print(shortdata)
#stemming of words
ps = PorterStemmer()
print('-------Stemming--------')
shortdata = shortdata.apply(lambda x: ' '.join([ps.stem(word) for word in x.split() ]))
print(shortdata)

#removing non-alphabets
shortdata=shortdata.apply(lambda x: ' '.join([word for word in x.split() if word.isalpha()]))

from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
#print(shortdata)
print('-------Lemmatization--------')
shortdata = shortdata.apply(lambda x: ' '.join([lmtzr.lemmatize(word,'v') for word in x.split() ]))
print(shortdata)


print('--------Removing punctuations--------')
def clear_punctuation(s):
	import string
	#print("\n")
	clear_string = ""
	for symbol in s:
		if symbol not in string.punctuation:
			clear_string += symbol
	return clear_string

shortdata = shortdata.apply(lambda x: ''.join(clear_punctuation(x))  ) 
for line in shortdata:
	print(line)
	print('\n')

def strip_all_entities(text):
	import string
	entity_prefixes = ['@']
	for separator in  string.punctuation:
		if separator not in entity_prefixes :
			text = text.replace(separator,' ')
	words = []
	for word in text.split():
		word = word.strip()
		if word:
			if word[0] not in entity_prefixes:
				words.append(word)
	return ' '.join(words)

shortdata = shortdata.apply(lambda x: ''.join(strip_all_entities(x))  ) 
print(shortdata)

#computing tfidf
trainset=shortdata.iloc[0:3]
print('-------Train set-------')
print(trainset)
testset=shortdata.iloc[3:5]
print('--------Test set-------')
print(testset)

# Create a set of frequent words
stoplist = set('for a of the and to in'.split(' '))
# Lowercase each document, split it by white space and filter out stopwords
texts = [[word for word in document.lower().split() if word not in stoplist] for document in trainset]
#print(texts)
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
#print(processed_corpus)

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora
dictionary = corpora.Dictionary(processed_corpus)
print('\n')
print(dictionary)
print('\n')
print(dictionary.token2id)
print('\n')
#word counts
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]


print('------Token ID and thier tf-idf weightings---------')
from gensim import models
# train the model
tfidf = models.TfidfModel(bow_corpus)
# transform the testset 
for text in testset:
	print(tfidf[dictionary.doc2bow(text.lower().split())])
	print('\n')
