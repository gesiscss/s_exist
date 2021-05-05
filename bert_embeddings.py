import numpy as np
import pandas as pd
from utils import build_feature_path, read_train, read_test

def load(df, embedding_file, embedding_type = 'cls', model = "binary", data_type = 'TRAINING_REL'):
	embeddings = np.load(embedding_file)
	print(len(embeddings))
	print(np.shape(embeddings))

	if embedding_type == 'cls': #only save the first layer
		doc_vectors = embeddings[:,0]
		print(len(doc_vectors[0]))
		df = pd.DataFrame(doc_vectors, index = df.index)
		print(df.head())
	elif embedding_type == 'avg_all_but_first': # take an average of the rest
		doc_vectors = np.mean(embeddings[:,1:], axis = 1)
		df = pd.DataFrame(doc_vectors, index = df.index)
	else:
		doc_vectors = np.mean(embeddings, axis = 1)
		df = pd.DataFrame(doc_vectors, index = df.index)
	
	rel_path = build_feature_path(data_type, 'bert_%s_%s' %(embedding_type, model))
	df.loc[df.index].to_csv(rel_path)




if __name__ == "__main__":
	train = read_train()
	test = read_test()

	data_types = [(test, 'TEST'),
				  (train, 'TRAINING')]

	for data, data_type in data_types:
		for embedding_type in ['cls', 'avg_all', 'avg_all_but_first']:
			load(data, "outputs/train_no_validation/%s_embeddings.npy" %(data_type), embedding_type = embedding_type,
			data_type = data_type + "_REL")

			load(data, "outputs/train_no_validation_multiclass/%s_embeddings.npy" %(data_type), embedding_type = embedding_type,
			data_type = data_type + "_REL", model = "multiclass")