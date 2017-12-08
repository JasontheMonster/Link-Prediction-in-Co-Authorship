'''building graph from dataset using networkx'''
import networkx as nx

class Graph(object):

	def __init__(self, edges, author, w, trainingSet, testingSet):
		self.train = nx.Graph()
		self.test = nx.Graph()
		for (node1, node2), year in edges:
			weightEdge = 1 / w[(node1, node2)]
			if year >=trainingSet[0] and year <=trainingSet[1]:
				self.train.add_edge(node1, node2, weight = weightEdge )
			elif year >=testingSet[0] and year <=testingSet[1]:
				self.test.add_edge(node1, node2, weight = weightEdge)

		self.train = max(nx.connected_component_subgraphs(train), key=len)


