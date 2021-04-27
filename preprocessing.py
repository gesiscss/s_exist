import re
import string

from bs4 import BeautifulSoup
from functools import partial

import spacy

__parser = None
spacy_stopwords = None  # depends on the parser, should `load_spacy` before use


def load_spacy(model_name='en_core_web_lg'):
    global __parser
    global spacy_stopwords
    if __parser is None:
        __parser = spacy.load(model_name)
        spacy_stopwords = __parser.Defaults.stop_words
        spacy_stopwords.update(set(string.punctuation))
        # import en_core_web_lg
        # parser = en_core_web_lg.load()


def get_parser():
    global __parser
    load_spacy()
    return __parser


emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE | re.I | re.M | re.DOTALL)
url_pattern = re.compile(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?')
mention_pattern = re.compile(r'(^|\W)(?P<mention>(rt|ht|cc|[.] ?)?(@\w+|MENTION\d+))(\b|[' + string.punctuation + '])',
                             flags=re.I | re.M | re.DOTALL)
hashtag_pattern = re.compile(r'(^|\W)(?P<hashtag>#\w+)(\b|[' + string.punctuation + '])', flags=re.I | re.M | re.DOTALL)

escape_punct_re = re.compile('[%s]' % re.escape(string.punctuation))


def replace_pattern(text, substitution, pattern, pattern_name):
    matched_spans = list()
    for match in re.finditer(pattern, text):
        matched_spans.append(match.span(pattern_name))
    new_text = substitution
    old_end = 0
    for start, end in matched_spans:
        new_text += text[old_end:start]
        old_end = end
    new_text += text[old_end:]
    return new_text


def detweet(text):  # TODO: might want to segment hashtags and mentions, or extract separate features
    text = re.sub(emoji_pattern, '', text)
    text = replace_pattern(text, '', hashtag_pattern, 'hashtag')
    text = replace_pattern(text, '', mention_pattern, 'mention')
    text = re.sub(url_pattern, '', text)
    return text


def normalize(text):
    return re.sub(r"\s+", " ",  # remove extra spaces
                  text).strip()


def unescape(text):
    return BeautifulSoup(text, features="html.parser").get_text()


def preprocess(text, fix_encoding=False):
    """
    It removes usernames, emoticons, hyperlinks/URLs and RT tag.
    params:
        tweets: list of unicode strings

    returns: list of unicode strings
    """
    if type(text) == str:
        if fix_encoding:
            return normalize(detweet(unescape(text)))
        else:
            return normalize(detweet(text))
    else:
        return text


def preprocess_light(tweets, fix_encoding=False):
    def _preprocess_light(text, fix_encoding=False):
        if type(text) == str:
            if fix_encoding:
                return normalize(re.sub(url_pattern, '', unescape(text)))
            else:
                return normalize(re.sub(url_pattern, '', text))
        else:
            return text

    return map(partial(_preprocess_light, fix_encoding=fix_encoding), tweets)


def escape_punct(x):
    return escape_punct_re.sub(' ', x)


def preprocess_for_embedding(text, model_name='en_core_web_lg'):
    load_spacy(model_name=model_name)

    text = preprocess(text, fix_encoding=True).lower()

    text = ' '.join(doc2token(text, remove_pron=False))
    return text


def doc2token(txt, remove_punct=True, remove_digit=True, remove_stops=True, remove_pron=True, lemmatize=True):
    parser = get_parser()
    parsed = parser(txt)
    tokens = list()
    for token in parsed:
        if remove_punct and token.is_punct:
            continue
        if remove_digit and token.is_digit:  # skip digits
            continue
        if remove_stops and (token.lemma_ in spacy_stopwords):  # skip stopwords
            continue
        if remove_pron and (token.lemma_ == '-PRON-'):  # skip pronouns
            continue
        else:
            token = token.lemma_.lower() if lemmatize else token.orth_.lower()
            if remove_punct:
                token = escape_punct(token)

            tokens.append(token.strip())
    return tokens


if __name__ == '__main__':
    from utils import read_sexism

    train = read_sexism()
    for text in train.tail(100).text:
        print(text)
        print(preprocess(text, fix_encoding=True))
        print("*" * 30)
