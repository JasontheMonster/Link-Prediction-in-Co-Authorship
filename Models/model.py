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
from sklearn.neighbors import KNeighborsClassifier
import time
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
class Model(object):

	def __init__(self, train_data, train_labels, test_data, test_labels):
		self.train_data = train_data
		self.train_labels = train_labels
		self.test_data = test_data
		self.test_labels = test_labels


	def random_forest(self):

		#15 6

		#20 7
		clf = RandomForestClassifier(n_estimators= 20, max_depth=7, random_state=0, min_samples_leaf=20)
		#clf = svm.SVC()
		clf.fit(self.train_data, self.train_labels)
		y_pred = []
		for query in tqdm(self.test_data):
			y_pred.append(clf.predict([query])[0])

		print (confusion_matrix(self.test_labels, y_pred))
		return clf.feature_importances_




	def svm_kernel(self):

		clf = svm.SVC(random_state = 0, verbose = True)
		#clf = svm.SVC()
		clf.fit(self.train_data, self.train_labels)
		y_pred = []
		for query in tqdm(self.test_data):
			y_pred.append(clf.predict([query])[0])

		print (confusion_matrix(self.test_labels, y_pred))


	def adaBoost(self):

		clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                         algorithm="SAMME",
                         n_estimators=20)
		#clf = svm.SVC()
		clf.fit(self.train_data, self.train_labels)
		y_pred = []
		for query in tqdm(self.test_data):
			y_pred.append(clf.predict([query])[0])

		print (confusion_matrix(self.test_labels, y_pred))

	def neuralNetwork(self):

		start = time.time()
		clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
		#clf = svm.SVC()
		clf.fit(self.train_data, self.train_labels)
		y_pred = []
		for query in self.test_data:
			y_pred.append(clf.predict([query])[0])

		print (time.time() - start)
		print (confusion_matrix(self.test_labels, y_pred))

	def knn(self):
		clf = KNeighborsClassifier(n_neighbors=3)
		#clf = svm.SVC()
		clf.fit(self.train_data, self.train_labels)
		y_pred = []
		for query in tqdm(self.test_data):
			y_pred.append(clf.predict([query])[0])

		print (confusion_matrix(self.test_labels, y_pred))


if __name__ == "__main__":
	test_labels = pickle.load( open ("test_labels.p", "rb"))
	test_data = pickle.load( open ("X_scaled_test.p", "rb"))
	train_labels = pickle.load( open ("labels.p", "rb"))
	train_data = pickle.load( open ("X_scaled_train.p", "rb"))

	#model = Model(train_data, train_labels, test_data, test_labels)
	#model.neuralNetwork()

	positive = []
	negative = []
	for train, label in zip(train_data, train_labels):
		if label == 1:
			positive.append(train)
		else:
			negative.append(train)

	positive = np.array(positive)
	negative = np.array(negative)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(positive[:,0], positive[:,1], positive[:,2], c="green", label ="Has Edge")
	ax.scatter(negative[:,0], negative[:,1], negative[:,2], c="blue", label = "No Edge")
	plt.legend(loc = "upper left")
	ax.set_xlabel('P')
	ax.set_ylabel('K')
	ax.set_zlabel('F')
	ax.view_init(azim=60)
	fig.savefig("test.png")




