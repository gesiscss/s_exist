from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def train(data):
	# reference: https://github.com/ThilinaRajapakse/simpletransformers#a-quick-example

	# Optional model configuration
	model_args = ClassificationArgs(num_train_epochs=1)

	# Create a ClassificationModel
	model = ClassificationModel(
    	"roberta", "roberta-base", args=model_args, use_cuda = False
	)

	# Train the model
	model.train_model(data)

	return model


if __name__ == "__main__":
	
	# train roberta models on the following:
	# - shared task training data
	# - share task training data minus validation data
	# - icwsm original data
	# - icwsm original + counterfactual data

	from utils import read_train, read_test, read_sexism

	# data = read_train()
	# print(data.head())
	# data['labels'] = data['task1']
	# data['labels'] = data['labels'].map({'sexist' : 1, 'non-sexist' : 0})
	# model = train(data[['text', 'labels']])

	data = read_sexism()
	print(data.head())
	data['labels'] = data['sexist']
	data['labels'] = data['labels'].map({True : 1, False : 0})
#	model = train(data[['text', 'labels']])




