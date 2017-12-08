class Metrics(object):

	def __init__(self, train, test, path_feature = False):
		self.train = train
		self.test = test
		self.attributes = attribute = json.load( open("attributes.json"))
		if path_feature:
			self.shortest_path = dict(nx.floyd_warshall(train, weight = 'weight'))

	def path(self, node1, node2):
		return self.shortest_path[node1][node2]
	
	def sum_key_word_count(self, node1, node2):
		return len(self.attribute[node1]["keywords"]) + len(self.attribute[node2]["keywords"])

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
		node1_words = self.attribute[node1]["keywords"]
		node2_words = self.attribute[node2]["keywords"]

		hit = 0
		for word1 in node1_words:
			for word2 in node2_words:
				if word1 == word2:
					hit+=1
		return hit

	#degree of matching for fields
	def fields_match(self, node1, node2):
		node1_words = self.attribute[node1]["fields"]
		node2_words = self.attribute[node2]["fields"]

		hit = 0 
		for i in range(len(node1_words)):
			for j in range(i+1, len(node2_words)):
				if node1_words[i] == node2_words[j]:
					hit+=1
		return hit