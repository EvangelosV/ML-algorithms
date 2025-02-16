import math
from collections import defaultdict
from preprocessing import tokenize

def build_vocabulary(texts, labels, n, k, m):

    doc_counts = defaultdict(int)
    pos_counts = defaultdict(int)
    neg_counts = defaultdict(int)
    num_docs = len(texts)
    
    for text, label in zip(texts, labels):
        words = set(tokenize(text))
        for word in words:
            doc_counts[word] += 1
            if label == 1:
                pos_counts[word] += 1
            else:
                neg_counts[word] += 1

    sorted_words = sorted(doc_counts.items(), key=lambda x: x[1], reverse=True)
    top_n_words = {w for w, count in sorted_words[:n]}
    sorted_words_asc = sorted(doc_counts.items(), key=lambda x: x[1])
    bottom_k_words = {w for w, count in sorted_words_asc[:k]}
    candidate_words = set(doc_counts.keys()) - top_n_words - bottom_k_words

    pos_total = sum(1 for l in labels if l == 1)
    p_pos = pos_total / num_docs
    def entropy(p):
        if p == 0 or p == 1:
            return 0
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)
    
    H_C = entropy(p_pos)
    info_gain = {}
    for word in candidate_words:
        docs_with = doc_counts[word]
        docs_without = num_docs - docs_with
        pos_with = pos_counts[word]
        p_pos_given_word = pos_with / docs_with if docs_with > 0 else 0
        H_C_given_word = entropy(p_pos_given_word)
        pos_without = pos_total - pos_with
        p_pos_given_not_word = pos_without / docs_without if docs_without > 0 else 0
        H_C_given_not_word = entropy(p_pos_given_not_word)
        H_C_given_feature = (docs_with / num_docs) * H_C_given_word + (docs_without / num_docs) * H_C_given_not_word
        ig = H_C - H_C_given_feature
        info_gain[word] = ig

    selected_words = sorted(info_gain.items(), key=lambda x: x[1], reverse=True)[:m]
    vocabulary = {word: idx for idx, (word, ig) in enumerate(selected_words)}
    return vocabulary

def text_to_feature_vector(text, vocabulary):

    words = set(tokenize(text))
    vector = [1 if word in words else 0 for word in vocabulary.keys()]
    return vector

def transform_texts(texts, vocabulary):

    return [text_to_feature_vector(text, vocabulary) for text in texts]
