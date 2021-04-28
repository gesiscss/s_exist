# s_exist
exist task at IberLEF


### to install
python requirements and resources:
```
pip install -r requirements.txt
python -m nltk.downloader punkt
python -m nltk.downloader stopwords
python -m nltk.downloader vader_lexicon
python -m spacy download en_core_web_lg
python -m spacy download es_core_news_lg
``` 

### troubleshoot
in case you have problems with relative imports, run:
```
export PYTHONPATH="${PYTHONPATH}:/path/to/your/project/
```