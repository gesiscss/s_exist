from functools import partial

import gensim.downloader as api
import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing import preprocess, doc2token
from utils import read_sexism, read_train, read_test, build_feature_path

if __name__ == '__main__':
    scales = read_sexism()
    scales = scales[scales.dataset == 'scales']
    train = read_train()
    test = read_test()
    print('preprocessing')
    scales['tokens'] = scales.text.apply(partial(preprocess, fix_encoding=True)).apply(doc2token)
    train['tokens'] = train.text.apply(partial(preprocess, fix_encoding=True)).apply(doc2token)
    test['tokens'] = test.text.apply(partial(preprocess, fix_encoding=True)).apply(doc2token)

    model = api.load('word2vec-google-news-300')


    def most_similar(text_tokens, scales, model):
        similarities = [model.wmdistance(text_tokens, scale_tokens) for scale_tokens in scales.tokens]
        scale_idx = np.argmin(similarities)
        return {'scale': scales.iloc[scale_idx].scale, 'sexist': scales.iloc[scale_idx].sexist,
                'sexist_content': scales.iloc[scale_idx].sexist_content}


    for dataset, dataset_key in [(train, 'TRAINING_REL'),
                                 (test, 'TEST_REL')]:
        print('processing', dataset_key)
        feature_store_path = build_feature_path(dataset_key, 'most_similar_scale')

        df = pd.DataFrame([most_similar(tokens, scales=scales, model=model) for tokens in tqdm(dataset.tokens)],
                          index=dataset.index)
        print(df.head())
        df.to_csv(feature_store_path)
