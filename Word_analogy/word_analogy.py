from sklearn.metrics.pairwise import cosine_similarity
def word_analogyy(dict,words):
    for w in words:
        w=w.strip()
        w=w.lower()
    wa=dict[words[0]]
    wb=dict[words[1]]
    wc=dict[words[2]]
    max_sim=-100
    try:
        for w in dict.vocab.keys():
            if w in words:
                continue
            wv = dict[w]
            sim = cosine_similarity([wb-wa],[wv-wc])
            if sim > max_sim:
                max_sim = sim
                d=w
    except :
        for w in dict.keys():
            if w in words:
                continue
            wd=dict[w]
            sim=cosine_similarity([wb-wa],[wd-wc])
            if sim>max_sim:
                max_sim=sim
                d=w
    return d