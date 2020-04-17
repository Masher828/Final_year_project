import gensim
import numpy as np
from gensim.models import word2vec
from gensim.models import keyedvectors
from sklearn.metrics.pairwise import cosine_similarity



def externalodd_out(word_vector,words):

    words=words.split(",")
    words=[w.strip() for w in words]
    all_word_vector=[word_vector[w.lower()] for w in words]
    avg_vector=np.mean(all_word_vector,axis=0)
    odd_one=None
    min_simi=100
    for w in words:

        # words=words.strip()
        sim=cosine_similarity([word_vector[w.lower()]],[avg_vector])
        if sim<min_simi:
            min_simi=sim
            odd_one=w
    return  odd_one
