#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Based on https://github.com/ddemszky/framing-twitter/blob/master/2_topic_clustering/3_tweet_embeddings.py
import random
import gc
import numpy as np
from sklearn.decomposition import TruncatedSVD


def get_word_weights(unigram_counts, a=1e-3):
    total_count = np.sum(list(unigram_counts.values()))
    unigram_weights = {word: (a / (a + word_freq / total_count)) for word, word_freq in unigram_counts.items()}
    # dict with words and their weights
    return unigram_weights


def get_weighted_average(We, x, m, w, dim):
    """
    Compute the weighted average vectors
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in sentence i
    :param w: w[i, :] are the weights for the words in sentence i
    :return: emb[i, :] are the weighted average vector for sentence i
    """
    print('Getting weighted average...')
    n_samples = x.shape[0]
    print(n_samples, dim)
    emb = np.zeros((n_samples, dim)).astype('float32')

    for i in range(n_samples):
        if i % 100000 == 0:
            print(i)
        stacked = []
        for idx, j in enumerate(x[i, :]):
            if m[i, idx] != 1:
                stacked.append(np.zeros(dim))
            else:
                stacked.append(We[j, :])
        vectors = np.stack(stacked)
        # emb[i,:] = w[i,:].dot(vectors) / np.count_nonzero(w[i,:])
        nonzeros = np.sum(m[i, :])
        emb[i, :] = np.divide(w[i, :].dot(vectors), np.sum(m[i, :]), out=np.zeros(dim),
                              where=nonzeros != 0)  # where there is a word
    del x
    del w
    gc.collect()
    return emb


def compute_pc(X, npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    print('Computing principal components...')
    svd.fit(X)
    return svd.components_


def sentences2idx(sentences, words2index, words2weight):
    """
    Given a list of sentences, output array of word indices that can be fed into the algorithms.
    :param sentences: a list of sentences
    :param words: a dictionary, words['str'] is the indices of the word 'str'
    :return: x1, m1. x1[i, :] is the word indices in sentence i, m1[i,:] is the mask for sentence i (0 means no word at the location)
    """
    # print(sentences[0].split())
    maxlen = min(max([len(word_tokenize(s)) for s in sentences]), 100)
    print('maxlen', maxlen)
    n_samples = len(sentences)
    print('samples', n_samples)
    x = np.zeros((n_samples, maxlen)).astype('int32')
    w = np.zeros((n_samples, maxlen)).astype('float32')
    x_mask = np.zeros((n_samples, maxlen)).astype('float32')
    for idx, s in enumerate(sentences):
        if idx % 100000 == 0:
            print(idx)
        split = word_tokenize(s)
        # split = s.split()
        indices = []
        weightlist = []
        for word in split:
            if word in words2index:
                indices.append(words2index[word])
                if word not in words2weight:
                    weightlist.append(0.000001)
                else:
                    weightlist.append(words2weight[word])
        length = min(len(indices), maxlen)
        x[idx, :length] = indices[:length]
        w[idx, :length] = weightlist[:length]
        x_mask[idx, :length] = [1.] * length
    del sentences
    gc.collect()
    return x, x_mask, w


def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    print('Removing principal component...')
    pc = compute_pc(X, npc)
    if npc == 1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


def SIF_embedding(We, x, m, w, rmpc, dim):
    """
    Compute the scores between pairs of sentences using weighted average + removing the projection on the first principal component
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in the i-th sentence
    :param w: w[i, :] are the weights for the words in the i-th sentence
    :param params.rmpc: if >0, remove the projections of the sentence embeddings to their first principal component
    :return: emb, emb[i, :] is the embedding for sentence i
    """
    emb = np.nan_to_num(get_weighted_average(We, x, m, w, dim))
    if rmpc > 0:
        emb = remove_pc(emb, rmpc)
    return emb


def get_document_embeddings(all_data, model, words2idx, dim, unigram_counts, rmpc=1):
    """
    :param docs: list of strings (i.e. docs), based on which to do the tf-idf weighting.
    :param all_data: dataframe column / list of strings (all tweets)
    :param model: pretrained word vectors
    :param vocab: a dictionary, words['str'] is the indices of the word 'str'
    :param dim: dimension of embeddings
    :param rmpc: number of principal components to remove
    :return:
    """
    print(dim)

    print('Getting word weights...')
    word2weight = get_word_weights(unigram_counts)
    # load sentences
    print('Loading sentences...')
    x, m, w = sentences2idx(all_data, words2idx,
                            word2weight)  # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
    print('Creating embeddings...')
    return SIF_embedding(model, x, m, w, rmpc, dim)  # embedding[i,:] is the embedding for sentence i


if __name__ == "__main__":
    from pmi import get_word_embeddings
    from nltk import word_tokenize
    from utils import read_train, read_test, build_feature_path
    import pandas as pd
    from preprocessing import preprocess_for_embedding

    RNG = random.Random()
    RNG.seed(42)

    print('reading data')
    train = read_train(languages=['en'])
    test = read_test(languages=['en'])
    df = pd.concat((train, test))

    print('preprocessing')
    df['text'] = df.text.apply(preprocess_for_embedding)
    docs = df.text.values
    print('tokenizing')
    docs = list(map(word_tokenize, docs))
    print('generate word embeddings')
    d = 50  # dimension of word embeddings (and hence, document embeddings)
    unigram_counts, id2tok, tok2id, word_vectors = get_word_embeddings(docs, embedding_size=d, window_length=3)
    print('generate document embeddings')
    doc_vectors = get_document_embeddings(df.text.values, word_vectors, tok2id, d, unigram_counts)
    print(len(train), len(test), doc_vectors.shape)

    sif_df = pd.DataFrame(doc_vectors, index=df.index)
    sif_training_rel_path = build_feature_path('TRAINING_REL', 'sif')
    sif_df.loc[train.index].to_csv(sif_training_rel_path)

    sif_test_rel_path = build_feature_path('TEST_REL', 'sif')
    sif_df.loc[test.index].to_csv(sif_test_rel_path)
