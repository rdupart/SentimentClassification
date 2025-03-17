import pickle
import os
import re
from sklearn.feature_extraction.text import HashingVectorizer

cur_dir = os.path.dirname(__file__)
stop = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'stopwords.pkl'), 'rb'))

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    text = re.sub(r'\W+', ' ', text.lower())
    return [word for word in text.split() if word not in stop]

vect = HashingVectorizer(decode_error='ignore', n_features=2**21, preprocessor=None, tokenizer=tokenizer)
