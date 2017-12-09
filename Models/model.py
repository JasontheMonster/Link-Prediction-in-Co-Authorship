from features import Metrics
import networkx as nx
import pickle
from tqdm import tqdm
from sklearn import svm
from sklearn.metrics import confusion_matrix
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from matplotlib import pyplot as plt

class Model(object):

	def __init__(self, train_data, train_labels, test_data, test_labels):
		self.train_data = train_data
		self.train_labels = train_labels
		self.test_data = test_data
		self.test_labels = test_labels


	def random_forest(self):

		clf = RandomForestClassifier(n_estimators= 30, max_depth=3, random_state=0, min_samples_leaf=10)
		#clf = svm.SVC()
		clf.fit(self.train_data, self.train_labels)
		y_pred = []
		for query in tqdm(self.test_data):
			y_pred.append(clf.predict([query])[0])

		print (confusion_matrix(self.test_labels, y_pred))
		return clf.feature_importances_


	def decision_Tree(self):

		clf = RandomForestClassifier(max_depth=2, random_state=0)
		#clf = svm.SVC()
		clf.fit(self.train_data, self.train_labels)
		print(clf.feature_importances_)
		y_pred = []
		for query in self.test_data:
			y_pred.append(clf.predict([query])[0])

		print (confusion_matrix(self.test_labels, y_pred))


	def svm_kernel(self):

		clf = svm.SVC(verbose = True)
		#clf = svm.SVC()
		clf.fit(self.train_data, self.train_labels)
		y_pred = []
		for query in tqdm(self.test_data):
			y_pred.append(clf.predict([query])[0])

		print (confusion_matrix(self.test_labels, y_pred))

	def svm(self):

		clf = RandomForestClassifier(max_depth=2, random_state=0)
		#clf = svm.SVC()
		clf.fit(self.train_data, self.train_labels)
		print(clf.feature_importances_)
		y_pred = []
		for query in self.test_data:
			y_pred.append(clf.predict([query])[0])

		print (confusion_matrix(self.test_labels, y_pred))

	def adaBoost(self):

		clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                         algorithm="SAMME",
                         n_estimators=200)
		#clf = svm.SVC()
		clf.fit(self.train_data, self.train_labels)
		y_pred = []
		for query in tqdm(self.test_data):
			y_pred.append(clf.predict([query])[0])

		print (confusion_matrix(self.test_labels, y_pred))

	def neuralNetwork(self):

		clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
		#clf = svm.SVC()
		clf.fit(self.train_data, self.train_labels)
		y_pred = []
		for query in self.test_data:
			y_pred.append(clf.predict([query])[0])

		print (confusion_matrix(self.test_labels, y_pred))


if __name__ == "__main__":
	test_labels = pickle.load( open ("test_labels.p", "rb"))
	test_data = pickle.load( open ("test_data.p", "rb"))
	train_labels = pickle.load( open ("labels.p", "rb"))
	train_data = pickle.load( open ("data_matrix.p", "rb"))
	model = Model(train_data, train_labels, test_data, test_labels)
	importance = model.random_forest()
	fig = plt.figure()

	y_pos = range(len(importance))
	objects = ["P", "CN", "K",
	 "F", "SK", "SN"]
	plt.bar(y_pos, importance, align='center', alpha=0.5)
	plt.xticks(y_pos, objects)
	plt.ylabel('Importance Score')
	plt.title('Different Features')
	fig.savefig("importances3.png")




