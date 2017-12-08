from tqdm import tqdm
import pickle
from graph import Graph
import json
import networkx as nx
class Metrics(object):

	def __init__(self, train, test, path_feature = False):
		self.train = train
		self.test = test
		self.info = json.load( open("attributes.json"))
		if path_feature:
			self.shortest_path = dict(nx.floyd_warshall(train, weight = 'weight'))

	def path(self, node1, node2):
		return self.shortest_path[node1][node2]
	
	def sum_key_word_count(self, node1, node2):
		return len(self.info[node1]["keywords"]) + len(self.info[node2]["keywords"])

	def sum_neighbors_count(self, node1, node2):
		return len(list(self.train.neighbors(node1))) + len(list(self.train.neighbors(node2)))

	#number of common_neighbors
	def common_neighbors(self, node1, node2):
		node1_neighbors = set()
		node2_neighbors = set()


		for n1 in self.train.neighbors(node1):
			node1_neighbors.add(n1)


		for n2 in self.train.neighbors(node2):
			node2_neighbors.add(n2)


		#print([node for node in node1_neighbors if node in node2_neighbors])
		return len(node1_neighbors.intersection(node2_neighbors))

	#degree of matching key_words
	def key_words_match(self, node1, node2):
		node1_words = self.info[node1]["keywords"]
		node2_words = self.info[node2]["keywords"]

		hit = 0
		for word1 in node1_words:
			for word2 in node2_words:
				if word1 == word2:
					hit+=1
		return hit

	#degree of matching for fields
	def fields_match(self, node1, node2):
		node1_words = self.info[node1]["fields"]
		node2_words = self.info[node2]["fields"]

		hit = 0 
		for i in range(len(node1_words)):
			for j in range(i+1, len(node2_words)):
				if node1_words[i] == node2_words[j]:
					hit+=1
		return hit


if __name__ == "__main__":
	dataGraph = pickle.load(open("graph_data.p", "rb"))
	metric = Metrics(dataGraph.train, dataGraph.test, path_feature = True)
	data = []
	labels = []
	test_data = []
	test_labels = []
	index = 0
	for node1 in tqdm(dataGraph.train):
		for node2 in dataGraph.train:
			if (node1, node2) not in dataGraph.train.edges and node1 != node2:
				#calculate path
				path_score = metric.path(node1, node2)

				#neighbors 
				neighbor_score = metric.common_neighbors(node1, node2)
				key_score = metric.key_words_match(node1, node2)
				field_score = metric.fields_match(node1, node2)
				kSum = metric.sum_key_word_count(node1, node2)
				nSum = metric.sum_neighbors_count(node1, node2)


				if index <= len(dataGraph.train.nodes)**2.0 - 2*len(dataGraph.train.edges):
					data.append([path_score, neighbor_score, key_score, field_score, kSum, nSum])
					labels.append(int((node1, node2) in dataGraph.test.edges))
				else:
					test_data.append([path_score, neighbor_score, key_score, field_score, kSum, nSum])
					test_labels.append(int((node1, node2) in dataGraph.test.edges))
			index +=1

	pickle.dump( data, open("data_matrix.p", "wb"))
	pickle.dump( labels, open("labels.p", "wb"))
	pickle.dump( test_data, open("test_data.p", "wb"))
	pickle.dump( test_labels, open("test_labels.p", "wb"))
