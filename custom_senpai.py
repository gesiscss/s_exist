import json
from collections import Counter, defaultdict
from functools import partial

import numpy as np
import pandas as pd
from minisom import MiniSom
from scipy.sparse import csr_matrix
from spacy.matcher import PhraseMatcher
from spacy.tokens import Token
from spacy.util import filter_spans

from pmi import get_word_embeddings
from preprocessing import get_parser, preprocess, doc2token

from utils import read_sexism, read_male, read_female, build_feature_path, read_train, read_test

canon = {'MALE': read_male()+['he', 'him', 'his'],
         'FEMALE': read_female()+['she', 'her', 'hers'],
         }


def dependency_ngram_indices_iteratitve(doc, n_minus_1_grams, pos_blacklist=['NUM', "PUNCT", "SPACE"],
                                        filter_stopwords=False):
    to_return = set()
    if (n_minus_1_grams is None) or (len(n_minus_1_grams) == 0):
        for token in doc:
            if (token.pos_ not in pos_blacklist) and not (filter_stopwords and token.is_stop) \
                    and (token._.cluster is not None):
                to_return.add((token.i,))
    else:
        for n_minus_1_gram in n_minus_1_grams:
            for head_index in n_minus_1_gram:
                for child_token in doc[head_index].children:
                    if (child_token.pos_ not in pos_blacklist) and not (filter_stopwords and child_token.is_stop) \
                            and (child_token._.cluster is not None):
                        if child_token.i not in n_minus_1_gram:
                            to_return.add(tuple(sorted(list(n_minus_1_gram) + [child_token.i])))
    return to_return


def find_dependency_collocations_iterative(docs, min_n, max_n, alpha, min_freq, pos_blacklist=['NUM', "PUNCT", "SPACE"],
                                           filter_stopwords=False):
    # this apriori-like algorithm is more memory and cpu efficient
    # however it assumes that all subpatterns are collocations (frequent and probable)

    # initialization:
    ngram_counts = defaultdict(Counter)
    doc2pattern = defaultdict(Counter)
    pattern2id = dict()
    pattern_index = 0
    # need to compute all intermediate ngrams, even if min_n >1
    n_range = range(1, max_n + 1)

    sent2doc = dict()  # maps sentences back to documents
    sents = list()
    for doc_id, doc in enumerate(docs):
        for sent_id, sent in enumerate(doc.sents):
            sent2doc[sent_id] = doc_id
            sents.append(sent.as_doc(copy_user_data=True))
    unigram_probabilities = None
    # generate ngrams: mapping n: mapping sent_id: set(1-tuple of indices)
    sent2ngram_index = defaultdict(lambda: defaultdict(set))
    sent2ngram_index[0] = {sent_id: set() for sent_id in sent2doc.keys()}
    # for each n
    for n in n_range:
        # expand n-1-grams into n-grams
        n_minus_1_grams = sent2ngram_index[n - 1]
        for sent_id in n_minus_1_grams:
            sent = sents[sent_id]
            for ngram_indices in dependency_ngram_indices_iteratitve(sent,
                                                                     n_minus_1_grams[sent_id],
                                                                     pos_blacklist=pos_blacklist,
                                                                     filter_stopwords=filter_stopwords):

                sent2ngram_index[n][sent_id].add(ngram_indices)
                ngram = [sent[i] for i in ngram_indices]
                encoding = encode_ngram(ngram)
                if encoding not in pattern2id:
                    pattern_index += 1
                    pattern2id[encoding] = pattern_index
                encoding_id = pattern2id[encoding]
                doc2pattern[sent2doc[sent_id]][encoding_id] += 1
                ngram_counts[n][encoding] += 1

        # filter frequent and probable ngrams
        ngram_count = float(sum(ngram_counts[n].values()))
        ngram_probabilities = defaultdict(lambda: 0.,
                                          {ngram: count / ngram_count for ngram, count in ngram_counts[n].items()})
        if n == 1:
            unigram_probabilities = ngram_probabilities
        filtered_ngrams = set(filter(
            partial(score_ngram, p_ngram=ngram_probabilities,
                    p_unigram=unigram_probabilities,
                    alpha=alpha),
            (ngram for ngram in ngram_probabilities.keys()
             if (ngram_counts[n][ngram] >= min_freq))))
        sent2ngram_index[n] = {sent_id: set(ngram_indices for ngram_indices in ngram_indices_set if
                                            encode_ngram([sents[sent_id][i] for i in ngram_indices]) in filtered_ngrams)
                               for sent_id, ngram_indices_set in sent2ngram_index[n].items()}

    # return only maximal dependency ngrams
    filtered_ngrams_by_n = {n: [encode_ngram([sents[sent_id][i] for i in ngram_indices])
                                for sent_id, ngram_indices_set in sent2ngram_index_by_n.items()
                                for ngram_indices in ngram_indices_set
                                ]
                            for n, sent2ngram_index_by_n in sent2ngram_index.items()
                            if n in range(min_n, max_n + 1)}
    ns = sorted(filtered_ngrams_by_n.keys(), reverse=True)
    for n in ns:
        subgrams = set(tuple(subgram) for ngram in filtered_ngrams_by_n[n]
                       for partition in partitions(list(ngram)) for subgram in partition
                       if len(subgram) < n
                       )
        for subgram in subgrams:
            if len(subgram) in filtered_ngrams_by_n:
                if subgram in filtered_ngrams_by_n[len(subgram)]:
                    filtered_ngrams_by_n[len(subgram)].remove(subgram)
    final_ngrams = sorted(ngram for ngrams in filtered_ngrams_by_n.values() for ngram in ngrams)
    # compactify final ngram ids
    final_ngrams2id = {ngram: id for id, ngram in enumerate(final_ngrams)}

    old_id2new_id = {pattern2id[ngram]: new_id for ngram, new_id in final_ngrams2id.items()}
    doc2ngram = {doc: {old_id2new_id[old_id]: count for old_id, count in ngram_counts.items()
                       if old_id in old_id2new_id}
                 for doc, ngram_counts in doc2pattern.items()}
    return final_ngrams, doc2ngram


def dependency_ngram_indices(doc, n=2, pos_blacklist=['NUM', "PUNCT", "SPACE"], filter_stopwords=False):
    if n <= 0:
        raise ValueError('n <= 0')
    if n == 1:
        return {(token.i,) for token in doc if
                (token.pos_ not in pos_blacklist) and not (filter_stopwords and token.is_stop)
                and (token._.cluster is not None)}
    else:
        n_minus_1_grams = dependency_ngram_indices(doc, n - 1, pos_blacklist=pos_blacklist,
                                                   filter_stopwords=filter_stopwords)
        to_return = set()
        for n_minus_1_gram in n_minus_1_grams:
            for head_index in n_minus_1_gram:
                for child_token in doc[head_index].children:
                    if (child_token.pos_ not in pos_blacklist) and not (filter_stopwords and child_token.is_stop) \
                            and (child_token._.cluster is not None):
                        if child_token.i not in n_minus_1_gram:
                            to_return.add(tuple(sorted(list(n_minus_1_gram) + [child_token.i])))
        return to_return


def dependency_ngrams(doc, n=2, pos_blacklist=['NUM', "PUNCT", "SPACE"], filter_stopwords=False):
    ngram_indices = dependency_ngram_indices(doc, n, pos_blacklist=pos_blacklist, filter_stopwords=filter_stopwords)
    n_grams = [[doc[i] for i in ngram] for ngram in ngram_indices]
    return n_grams


def encode_token(token):
    return token._.cluster
    # if token._.canon:
    #     # print(token, token._.canon)
    #     return token._.canon
    # else:
    #     return token._.cluster
    #     # return token.lemma_.lower() + "_" + token.pos_


def encode_ngram(ngram):
    return tuple(sorted(encode_token(token) for token in ngram))


def encoded_dependency_ngrams(doc, n=2, pos_blacklist=['NUM', "PUNCT", "SPACE"], filter_stopwords=False):
    return map(encode_ngram, dependency_ngrams(doc, n, pos_blacklist=pos_blacklist, filter_stopwords=filter_stopwords))


def partitions(collection):
    # https://stackoverflow.com/a/30134039
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in partitions(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
        # put `first` in its own subset
        yield [[first]] + smaller


def score_ngram(ngram, p_ngram, p_unigram, alpha=2):
    # ideally
    # p (ngram) >= alpha max(product(p(i) for i in partition) for partition in proper_partitions)
    # p(i) ~ freq(i)/|(j, k)grams| is ill defined though, since there is some double-counting when mixing
    #       different ngrams
    # p(i) ~ freq(i)/||i|grams| could work, but ideally one would drop unigrams that get merged and recount
    #       probabilities. It could be doable computing p() for increasing n, and removing frequencies of all
    #       (sub-)ngrams in proper partitions of newly added ngrams
    # p(ngram) >= alpha max(product(p(i) for i in unigrams)) works but is simplistic
    if len(ngram) == 1:
        return True
    else:
        return p_ngram[ngram] >= alpha * np.product([p_unigram[(unigram,)] for unigram in ngram])


def find_dependency_collocations(docs, min_n, max_n, alpha, min_freq, pos_blacklist=['NUM', "PUNCT", "SPACE"],
                                 filter_stopwords=False):
    ngram_counts = defaultdict(Counter)
    n_range = [1] + list(range(max(min_n, 2), max_n + 1))

    doc2pattern = defaultdict(Counter)
    pattern2id = dict()
    pattern_index = 0
    for n in n_range:
        for doc_id, doc in enumerate(docs):
            for sent in doc.sents:
                for encoding in encoded_dependency_ngrams(sent.as_doc(copy_user_data=True), n,
                                                          pos_blacklist=pos_blacklist,
                                                          filter_stopwords=filter_stopwords):
                    if encoding not in pattern2id:
                        pattern_index += 1
                        pattern2id[encoding] = pattern_index
                    encoding_id = pattern2id[encoding]
                    doc2pattern[doc_id][encoding_id] += 1

                    ngram_counts[n][encoding] += 1

    total_counts = {n: sum(counts.values()) for n, counts in ngram_counts.items()}
    ngram_probabilities = {ngram: count / float(total_counts[n]) for n, counts in ngram_counts.items()
                           for ngram, count in counts.items()}
    filtered_ngrams = filter(
        partial(score_ngram, p_ngram=ngram_probabilities, p_unigram=ngram_probabilities, alpha=alpha),
        (ngram for ngram in ngram_probabilities.keys()
         if (ngram_counts[len(ngram)][ngram] >= min_freq) and (len(ngram) >= min_n)))
    # return only maximal dependency ngrams
    filtered_ngrams_by_n = defaultdict(list)
    for ngram in filtered_ngrams:
        filtered_ngrams_by_n[len(ngram)].append(ngram)
    ns = sorted(filtered_ngrams_by_n.keys(), reverse=True)
    for n in ns:
        subgrams = set(tuple(subgram) for ngram in filtered_ngrams_by_n[n]
                       for partition in partitions(list(ngram)) for subgram in partition
                       if len(subgram) < n
                       )
        for subgram in subgrams:
            if subgram in filtered_ngrams_by_n[len(subgram)]:
                filtered_ngrams_by_n[len(subgram)].remove(subgram)
    id2pattern = {id: pattern for pattern, id in pattern2id.items()}
    final_ngrams = sorted(ngram for ngrams in filtered_ngrams_by_n.values() for ngram in ngrams)
    doc2ngram = {doc: {final_ngrams.index(id2pattern[ngram]): count for ngram, count in ngram_counts.items()
                       if id2pattern[ngram] in final_ngrams
                       } for doc, ngram_counts in doc2pattern.items()}
    return final_ngrams, doc2ngram


def merge_on_matcher(doc, nlp, phrase_matcher):
    with doc.retokenize() as retokenizer:
        spans = list()
        span_labels = dict()
        for match_id, start, end in phrase_matcher(doc):
            spans.append(doc[start:end])
            span_labels[(start, end)] = nlp.vocab.strings[match_id]
            for token in doc[start:end]:
                token._.canon = nlp.vocab.strings[match_id]

        filtered_spans = filter_spans(spans)
        for span in filtered_spans:
            retokenizer.merge(span)
            for token in span:
                token._.canon = span_labels[(span.start, span.end)]

    return doc


def train_som(data, n_iterations, vector_size, map_side, sigma, learning_rate, random_seed):
    som = MiniSom(x=map_side,
                  y=map_side,
                  input_len=vector_size,
                  sigma=sigma,
                  learning_rate=learning_rate,
                  random_seed=random_seed)  # initialization of a square SOM
    som.train_random(data, n_iterations)  # trains the SOM
    return som


if __name__ == '__main__':
    print('get parser')
    nlp = get_parser()
    Token.set_extension("canon", default=None)
    Token.set_extension("cluster", default=None)

    print('read input')
    train = read_train()
    test = read_test()
    df = pd.concat((train, test))
    # df = read_sexism()
    print('preprocess')
    df['body'] = df.text.apply(partial(preprocess, fix_encoding=True))
    print('spacify')
    df['doc'] = df.body.apply(nlp)
    print('merge matches')
    patterns = {key: [nlp(phrase) for phrase in phrases] for key, phrases in canon.items()}
    phrase_matcher = PhraseMatcher(vocab=nlp.vocab, attr="LOWER")
    for k, p in patterns.items():
        phrase_matcher.add(k, p)

    docs = [merge_on_matcher(doc, nlp, phrase_matcher) for doc in df.doc]
    doc_index = df.reset_index().id.values.tolist()
    # spacy's Token.cluster is set to 0 in any model I use
    # cluster_map = defaultdict(Counter)
    # for doc in docs:
    #     for tok in doc:
    #         cluster_map[str(tok.cluster)][tok.orth_.lower()]+=1
    # print(len(cluster_map), 'clusters total')
    pos_blacklist = ['NUM', "PUNCT", "SPACE", "SYM"],
    filter_stopwords = True
    cluster_words = True

    if cluster_words:
        print('cluster')
        EMBEDDING_SIZE = 50
        unigram_counts, id2tok, tok2id, pmis = get_word_embeddings([doc2token(doc, pos_blacklist=pos_blacklist,
                                                                              remove_pron=False)
                                                                    for doc in docs],
                                                                   embedding_size=EMBEDDING_SIZE, window_length=3)
        som = train_som(pmis, 1000, vector_size=EMBEDDING_SIZE, map_side=32, sigma=0.3,
                        learning_rate=0.5,
                        random_seed=0)
        winner_indices = som.win_map(pmis, return_indices=True)
        cluster_map = defaultdict(lambda: None, {id2tok[i]: encoding for encoding, indices in winner_indices.items()
                                                 for i in indices
                                                 })

        cluster_namer = defaultdict(Counter)
        for doc in docs:
            for tok in doc:
                if (tok.pos_ not in pos_blacklist) and not (filter_stopwords and tok.is_stop):
                    lemma = tok.lemma_.lower()
                    if tok._.canon is not None:
                        cluster = tok._.canon
                    else:
                        cluster = str(cluster_map[lemma])
                    tok._.cluster = cluster
                    cluster_namer[cluster][lemma] += 1


        def format_ngram(ngram):
            to_return = list()
            for tok in ngram:
                if tok in canon:
                    to_return.append(tok.upper())
                else:
                    to_return.append(','.join([w for w, c in cluster_namer[tok].most_common(3)]))
            return '\t'.join(to_return)
    else:

        for doc in docs:
            for tok in doc:
                if (tok.pos_ not in pos_blacklist) and not (filter_stopwords and tok.is_stop):
                    lemma = tok.lemma_.lower()
                    if tok._.canon is not None:
                        cluster = tok._.canon
                    else:
                        cluster = lemma
                    tok._.cluster = cluster


        def format_ngram(ngram):
            to_return = list()
            for tok in ngram:
                if tok in canon:
                    to_return.append(tok.upper())
                else:
                    to_return.append(tok)
            return '\t'.join(to_return)

    print('mine')
    final_ngrams, doc2ngram = find_dependency_collocations(docs, min_n=1, max_n=4, alpha=2, min_freq=10,
                                                           pos_blacklist=pos_blacklist,
                                                           filter_stopwords=filter_stopwords
                                                           )
    # final_ngrams, doc2ngram = find_dependency_collocations_iterative(docs, min_n=2, max_n=4, alpha=2, min_freq=10,
    #                                                                  pos_blacklist=pos_blacklist,
    #                                                                  filter_stopwords=filter_stopwords
    #                                                                  )

    for ngram in final_ngrams:
        print(format_ngram(ngram))
    # build a doc x sparse pattern matrix with the following code
    row_ind = list()
    col_ind = list()
    data = list()
    for doc_idx, ngram_counts in doc2ngram.items():
        for ngram_index, ngram_count in ngram_counts.items():
            row_ind.append(doc_idx)
            col_ind.append(ngram_index)
            data.append(ngram_count)
    (M, N) = (len(docs), len(final_ngrams))
    bag_of_dep_ngrams = csr_matrix((data, (row_ind, col_ind)), shape=(M, N))

    print(M, N)

    senpai_df = pd.DataFrame(bag_of_dep_ngrams.todense(), index=doc_index)
    senpai_training_rel_path = build_feature_path('TRAINING_REL', 'senpai')
    senpai_df.loc[train.index].to_csv(senpai_training_rel_path)

    senpai_test_rel_path = build_feature_path('TEST_REL', 'senpai')
    senpai_df.loc[test.index].to_csv(senpai_test_rel_path)


    # senpai_training_rel_path = build_feature_path('TRAINING_REL', 'senpai')
    # with open(senpai_training_rel_path, 'w+') as f:
    #     json.dump(dict(final_ngrams=final_ngrams, doc2ngram=doc2ngram, doc_index=doc_index), f)
