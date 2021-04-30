from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def train(data, eval_df = None, savepath = "outputs/"):
	# reference: https://github.com/ThilinaRajapakse/simpletransformers#a-quick-example

	# Optional model configuration
	model_args = ClassificationArgs(num_train_epochs=1, output_dir = savepath)

	# Create a ClassificationModel
	model = ClassificationModel(
    	"roberta", "roberta-base", args=model_args, use_cuda = False
	)

	# Train the model
	model.train_model(data)

	# Evaluate the model
	if eval_df:
		result, model_outputs, wrong_predictions = model.eval_model(eval_df)
		print(result)

	# Make predictions with the model
	# predictions, raw_outputs = model.predict(["Sam was a Wizard"])

	return model


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


	for train_type in train_types:
		if train_type == 'train_no_validation':
			data = read_train_no_validation()
		
		print(data.head())
		data['labels'] = data[label_field[train_type]]
		data['labels'] = data['labels'].map(labels[train_type])
	
		model = train(data, test, savepath = "outputs/%s" %train_type)