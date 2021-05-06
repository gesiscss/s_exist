from simpletransformers.classification import ClassificationModel, ClassificationArgs
import numpy as np
import pandas as pd
import logging
import random

from sklearn.metrics import classification_report


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def predict(eval_df, test_name = "TEST", dataset_key = 'TEST_REL', loadpath = "outputs/",
			 task = 'task1', run = '1', label_mapping = {1 : 'sexist', 0 : 'non-sexist'}):

	from utils import generate_test_run, build_feature_path

	# testing with a random generator
	# eval_df['predictions'] = [random.randint(0, 1) for _ in range(len(eval_df))]
	
	# reference: https://github.com/ThilinaRajapakse/simpletransformers#a-quick-example

	# load model
	model = ClassificationModel(
	"roberta", loadpath, use_cuda = False
)

	# Make predictions with the model
	eval_df['predictions'], raw_outputs, all_embedding_outputs,\
	 all_layer_hidden_states = model.predict(list(eval_df['text'].values))

	print(np.shape(all_embedding_outputs[1]))
	print(len(all_embedding_outputs), len(all_embedding_outputs[0]),
	 len(all_embedding_outputs[-1][0]), all_embedding_outputs[1][0][0])
	print(type(all_embedding_outputs))

	with open(loadpath+"/%s_embeddings.npy" %(test_name), 'wb') as f:
		np.save(f, all_embedding_outputs)


	eval_df = eval_df.reset_index()
	eval_df['id'] = eval_df['id'].astype(str)
	eval_df['predictions'] = eval_df['predictions'].map(label_mapping)

	# save the labels as features
	feature_store_path = build_feature_path(dataset_key, 'bert_multiclass')
	df = eval_df[['id', 'predictions']]
	df.to_csv(feature_store_path, index = False)

	# save runs ONLY for test data
	if test_name == 'TEST':
		eval_df['binary_predictions'] = ['non-sexist' if i != 'sexist' else i for i in eval_df['predictions']]
		eval_df['binary_predictions'] = eval_df['predictions']
		eval_df = eval_df[['test_case', 'id', 'predictions']]
		generate_test_run(eval_df, task = task, run = "3", model_type = 'binarized_multiclass_bert')

	# for validation, print results
	if test_name == 'VALIDATION':
		print(classification_report(eval_df['task2'], eval_df['predictions']))
		# convert to binary and see performance
		eval_df['binary_predictions'] = ['non-sexist' if i != 'sexist' else i for i in eval_df['predictions']]
		print(classification_report(eval_df['task1'], eval_df['binary_predictions']))




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

	from utils import read_validation, read_train, read_test

	train = read_train()
	test = read_test()
	val = read_validation()

	label_mapping = dict(zip(range(0, len(train.task2.unique())),
								sorted(train.task2.unique())))

	for dataset, dataset_key in [
								 #(train, 'TRAINING_REL'),
								 (val, 'VALIDATION_REL'),
								 (test, 'TEST_REL'),]:

		print(dataset.head())
		predict(dataset, loadpath = '/bigdata/indira/s_exist/outputs/train_no_validation_multiclass/',
	 	test_name = dataset_key[:-4], dataset_key = dataset_key,
	 	 label_mapping = label_mapping, task = "task2")


