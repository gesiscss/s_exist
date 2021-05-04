from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def train(data, savepath = "outputs/", num_classes = 2):
	# reference: https://github.com/ThilinaRajapakse/simpletransformers#a-quick-example

	# Optional model configuration
	model_args = ClassificationArgs(num_train_epochs=5, output_dir = savepath)

	# Create a ClassificationModel
	model = ClassificationModel(
    	"roberta", "roberta-base", args=model_args, 
    	use_cuda = False, num_labels = num_classes
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

	train_types = ["train_no_validation", 
				   #"train",
				   #"validation"
				   ]

	label_field = {"train_no_validation" : "task2",
				   "train" : "task2",
				   "validation" : "task2"

				   }

	from utils import read_train, read_test, read_sexism, read_train_no_validation, read_validation

	data = read_validation()
		
	print(data.head())
	print(data.task2.unique())

	label_mapping = dict(zip(sorted(data.task2.unique()), range(0, len(data.task2.unique()))))

	print(label_mapping)

	for train_type in train_types:
		data = read_train_no_validation()
		
		print(data.head())
		print(data.task2.unique())

		label_mapping = dict(zip(sorted(data.task2.unique()), range(0, len(data.task2.unique()))))

		print(label_mapping)


		data['labels'] = data[label_field[train_type]]
		data['labels'] = data['labels'].map(label_mapping)

		# downsample non-sexist data
		print(data.groupby('labels').size())

		g = data.groupby('task2')
		print(g.get_group('non-sexist'))
		data = g.apply(lambda x: x.sample(700).reset_index(drop=True) \
			if x.labels.unique()[0] == 2 else x.sample(len(x)).reset_index(drop=True))

		print(data.groupby('labels').size())
	
		model = train(data, savepath = "outputs/%s_multiclass" %train_type, num_classes = len(data.task2.unique()))