import pandas as pd
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing import text
import os
# Moving up two directory path to get the path of raw data
os.chdir('..')
os.chdir('..')
# Path to raw input data
raw_data_dir = os.path.join(os.getcwd(), 'data', 'raw', 'Train_aO7sTW8')
# Reading the raw input data
df = pd.read_csv(os.path.join(raw_data_dir, 'Train.csv'))
# Initialise an empty list
sentences = []
for abst in df['ABSTRACT']:
    # Removing the filters from the text- '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    sentences.append(text.text_to_word_sequence(abst, lower=True))
# Word Embedding
model = Word2Vec(sentences, workers=8, size=100)
model.wv.save_word2vec_format(os.path.join(os.path.dirname(__file__),'Word2Vec_Hacklive3_100d.txt'), binary=False)
