# s_exist
exist task at IberLEF


### to install
python requirements and resources:
```shell
sudo apt-get install build-essential swig python3-dev
pip install -r requirements.txt
curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip3 install
pip3 install auto-sklearn
python -m nltk.downloader punkt
python -m nltk.downloader stopwords
python -m nltk.downloader vader_lexicon
python -m spacy download en_core_web_lg
``` 

### to run
download the data and create a configuration script called `config.json` in the root of the project. 
A sample configuration file is available as `config_sample.json`. You will need the following data:
- the EXIST2021 train and test datasets
- the list of male and female words
- the sexism dataset
- hedge, booster, discourse markers lexicons

Translate Spanish data into English with: 
```shell
translate spanish data.ipynb
```

Then, run the feature extraction scripts:
```shell
python custom_senpai.py
python sif_embeddings.py
python most_similar_scale.py
python feature_extraction_perspective.py
python vader.py
python gender_words.py
python lexica.py
python mentions_and_hashtags.py
```
Split the data in training and validation:
```shell
explore and split data.ipynb
```
Then feature selection:
```shell
python feature_selection.py
```
Model tuning and  classification:
* finetuned_classify*.py
* automl_task*.ipynb
* gridsearch_task*.ipynb


### troubleshoot
in case you have problems with relative imports, run:
```shell
export PYTHONPATH="${PYTHONPATH}:/path/to/your/project/
```