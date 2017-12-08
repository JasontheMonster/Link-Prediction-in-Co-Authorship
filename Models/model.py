from features import Metrics
class Model(object):

	def __init__(self, train, test):
		self.train = train
		self.test = test
		self.train_data = []
		self.train_labels = []
		self.test_data = []
		self.test_labels = []
		self.metrics = Metrics(train, test, path_feature = True)



	def data_matrix(self):  
		index = 0
		for node1 in tqdm(self.train.nodes):
			for node2 in self.train.nodes:
				if (node1, node2) not in self.train.edges and node1 != node2:
					#calculate path
					path_score = self.metrics.path(node1, node2)

					#neighbors 
					neighbor_score = self.metrics.common_neighbors(node1, node2)
					key_score = self.metrics.key_words_match(node1, node2)
					field_score = self.metrics.fields_match(node1, node2)
					kSum = self.metrics.sum_key_word_count(node1, node2)
					nSum = self.metrics.sum_neighbors_count(node1, node2)
					if index <= len(train.nodes)**2.0 - 2*len(train.edges):
						self.train_data.append([path_score, neighbor_score, key_score, field_score, kSum, nSum])
						self.train_labels.append(int((node1, node2) in self.test.edges))
					else:
						self.test_data.append([path_score, neighbor_score, key_score, field_score, kSum, nSum])
						self.test_labels.append(int((node1, node2) in self.test.edges))
				index +=1

	def random_forest(self):

		clf = RandomForestClassifier(max_depth=2, random_state=0)
		#clf = svm.SVC()
		clf.fit(self.train_data, self.train_labels)
		print(clf.feature_importances_)
		y_pred = []
		for query in self.test_data:
			y_pred.append(clf.predict([query])[0])

		print (confusion_matrix(self.test_labels, y_pred))