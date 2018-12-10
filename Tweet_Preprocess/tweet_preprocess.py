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

print(shortdata)

#removing stopwords
from nltk.corpus import stopwords
stop=stopwords.words("english")
print('------Removing stopwords------')
shortdata=shortdata.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
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

	
with open("tweets.csv","w") as file:	
	for i in range(5):															
		file.write(shortdata1[i])
		file.write("     ")
		file.write(shortdata[i])
		file.write('\n')


			
