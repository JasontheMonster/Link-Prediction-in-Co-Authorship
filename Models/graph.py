'''building graph from dataset using networkx'''

import networkx as nx
import pickle
import json

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

		self.train = max(nx.connected_component_subgraphs(self.train), key=len)


if __name__ == "__main__":
	edges = pickle.load( open("newEdges.p", "rb"))
	authors = pickle.load( open("newAuthor.p", "rb"))
	w = pickle.load( open("weights.p", "rb"))

	dataGraph = Graph(edges, authors, w, [1967, 2006], [2007, 2017])
	print (nx.info(dataGraph.train))
	pickle.dump(dataGraph, open('graph_data.p', "wb"))
