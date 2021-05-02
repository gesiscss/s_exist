from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging

from sklearn.metrics import classification_report


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def predict(eval_df, loadpath = "outputs/"):
	# reference: https://github.com/ThilinaRajapakse/simpletransformers#a-quick-example

	# load model
	model = ClassificationModel(
    "roberta", loadpath, use_cuda = False
)

	# Make predictions with the model
	predictions, raw_outputs = model.predict(list(eval_df['text'].values))

	print(classification_report(list(eval_df['labels'].values), predictions))



if __name__ == "__main__":
	
	# train roberta models on the following:
	# - shared task training data
	# - share task training data minus validation data
	# - icwsm original data
	# - icwsm original + counterfactual data

	train_types = ["train_no_validation", 
				   "sexism",
				   "train",
				   ]

	label_field = {"train_no_validation" : "task1",
				   "sexism" : "sexist",
				   "train" : "task1"
				   }

	labels = {"train_no_validation" : {'sexist' : 1, 'non-sexist' : 0},
				   "sexism" : {True : 1, False : 0},
				   "train" : {'sexist' : 1, 'non-sexist' : 0},
				   }

	from utils import read_train, read_test, read_sexism, read_train_no_validation, read_validation

	test = read_validation()
	print(test.head())
	test['labels'] = test[label_field['train_no_validation']]
	test['labels'] = test['labels'].map(labels['train_no_validation'])

	predict(test, loadpath = 'outputs/train_no_validation')
	
	# for train_type in train_types:
	# 	if train_type == 'train_no_validation':
	# 		data = read_train_no_validation()
		
	# 	print(data.head())
	# 	data['labels'] = data[label_field[train_type]]
	# 	data['labels'] = data['labels'].map(labels[train_type])
	
	# 	model = train(test, loadpath = "outputs/%s" %train_type)