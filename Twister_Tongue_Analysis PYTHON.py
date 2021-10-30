#Twister Tongue Corpus Analysis
#Natalia Wojarnik

#Useful libraries and modules

import nltk, re, pprint, string
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer

#Reading a raw text without cleaning

my_corpus = open("Tongue_Twister_Corpus.txt")
text = my_corpus.read()
#The text is loaded and it's a string as a Python object. Printing the first 100 entries would give the first 100 characters (letters, whitespaces, punctuation etc.).
text = text.split()
print(text[:20])

#Cleaning the data
def remove_punc(x):
    x_words = []
    for w in x:
        word = re.sub(r'[^\w\s\-\']','',w) #this removes everything except words, spaces, hyphens and apostrophes
        x_words.append(word)
    return x_words

text = remove_punc(text)
print(text[:20])

tokens = [w.lower() for w in text]
print(tokens[:20])
print(f"There are {len(tokens)} tokens in the corpus.")

types = sorted(set(tokens))
print(types[:20])
print(f"There are {len(types)} types in the corpus.")
types_perc = round(100*len(types)/len(tokens))
print(f"Percentage of unique words in the text is: {types_perc}%")

#Checking the frequency distribution over the types in the corpus.
freq_d = nltk.FreqDist(tokens)
x = 10
most_com = freq_d.most_common(x)
print(f"{x} most common words are: {most_com}\n")
print(f"Token that has the most types is: {most_com[0]}")
y = 40
freq_d.plot(y, cumulative = True, title = f"Frequency distribution of first {y} words")

#Frequency conclusion - Zipf's Law 
hapaxes = freq_d.hapaxes()
print(f"There are {len(hapaxes)} hapaxes in the corpus:\n")
hapaxes.sort()
print(hapaxes)
print(f"\nHapaxes determine {round(100*len(hapaxes)/len(types))}% of types in the corpus.")

#Lemmatization and Stemming
lemmatizer = WordNetLemmatizer()
lemm_tokens = ' '.join([lemmatizer.lemmatize(w) for w in tokens])
lemm_tokens = lemm_tokens.split()
print(lemm_tokens[:20])

stemmer = PorterStemmer()
def stemming(x):
    list_tokens = []
    for w in x:
        root = stemmer.stem(w)
        list_tokens.append(root)
    return list_tokens

stem_tokens = stemming(lemm_tokens)
print(stem_tokens[:20])
print(f"\nThere are {len(stem_tokens)} tokens in the corpus (using Lemmatizer and Stemmer).")

stem_types = sorted(set(stem_tokens))
print(stem_types[:20])
print(f"\nThere are {len(stem_types)} types in the corpus (using Lemmatizer and Stemmer).")

stem_types_perc = round(100*len(stem_types)/len(stem_tokens))
print(f"Percentage of unique words in the text (using Lemmatizer and Stemmer) is: {stem_types_perc}%")

#Frequency distribution using Lemmatizer and Stemmer
stem_freq_d = nltk.FreqDist(stem_tokens)
x = 10
stem_most_com = stem_freq_d.most_common(x)
print(f"{x} most common words are: {stem_most_com}\n")

print(stem_most_com == most_com)
print(stem_most_com)
print(most_com)
print(f"\nThe most {x} frequent words in the corpus didn't change after using Lemmatizer and Stemmer. But what is interesting, their counts changed.")
print(f"Token that has the most types is: {stem_most_com[0]}\nToken is the same comparing to manual lemmatization but the count is different.")

stem_hapaxes = stem_freq_d.hapaxes()
print(f"There are {len(stem_hapaxes)} hapaxes in the corpus (using Lemmatizer and Stemmer):\n")
stem_hapaxes.sort()
print(stem_hapaxes)
print(f"\nHapaxes determine {round(100*len(stem_hapaxes)/len(stem_types))}% of types in the corpus (using Lemmatizer and Stemmer).")
