import re
import string

from bs4 import BeautifulSoup
from functools import partial

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


def detweet(text): # TODO: might want to segment hashtags and mentions, or extract separate features
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


if __name__ == '__main__':
    from utils import read_train

    train = read_train()
    for text in train.tail(100).text:
        print(text)
        print(preprocess(text, fix_encoding=True))
        print("*" * 30)