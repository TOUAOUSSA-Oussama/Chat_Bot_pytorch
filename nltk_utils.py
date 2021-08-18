# Data preprocessing :
import nltk
import numpy as np
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

# 1/- Tokenization: convertir une phrase en une liste de mots
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# 2/- Stemming + lower case: prendre que la partie première utile d'un mot
stemmer = PorterStemmer()
def stem(word):
    return stemmer.stem(word.lower())

# 3/- bag_of_words : convertir une liste de mots en une liste de 0 et 1
def bag_of_words(tokenized_sentence, all_words):
    """
    :param tokenized_sentence:  ['hello', 'how', 'are', 'you']
    :param all_words:  ['hi', 'hello', 'i', 'you', 'bye', 'thank', 'cool']
    :return:  [0, 1, 0, 1, 0, 0, 0]
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence :
            bag[idx] = 1.0
    return bag