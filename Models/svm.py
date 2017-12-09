import networkx as nx
import pickle
from tqdm import tqdm
from sklearn import svm
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import json
from sklearn.ensemble import RandomForestClassifier

edges = pickle.load( open("newEdges.p", "rb"))
authors = pickle.load( open("newAuthor.p", "rb"))
attribute = json.load( open("attributes.json"))
w = pickle.load( open("weights.p", "rb"))

minYear = 100000
train = nx.Graph()
test = nx.Graph()
for (node1, node2), year in edges:
	weightEdge = 1 / w[(node1, node2)]
	if year >=1967 and year <=2006:
		train.add_edge(node1, node2, weight = weightEdge )
	elif year >=2007 and year <=2017:
		test.add_edge(node1, node2, weight = weightEdge)


print (nx.info(train))
print (nx.info(test))

train = max(nx.connected_component_subgraphs(train), key=len)




print (nx.info(train))




#shortest_path = dict(nx.floyd_warshall(train, weight = 'weight'))

#shortest path between two nodes
#def path(node1, node2):
#	return shortest_path[node1][node2]


def sum_key_word_count(node1, node2):
	return len(attribute[node1]["keywords"]) + len(attribute[node2]["keywords"])

def sum_neighbors_count(node1, node2):
	return len(list(train.neighbors(node1))) + len(list(train.neighbors(node2)))



#number of common_neighbors
def common_neighbors(node1, node2, train):
	node1_neighbors = set()
	node2_neighbors = set()


	for n1 in train.neighbors(node1):
		node1_neighbors.add(n1)


	for n2 in train.neighbors(node2):
		node2_neighbors.add(n2)


	#print([node for node in node1_neighbors if node in node2_neighbors])
	return len(node1_neighbors.intersection(node2_neighbors))

#degree of matching key_words
def key_words_match(node1, node2):
	node1_words = attribute[node1]["keywords"]
	node2_words = attribute[node2]["keywords"]

	hit = 0
	for word1 in node1_words:
		for word2 in node2_words:
			if word1 == word2:
				hit+=1
	return hit

#degree of matching for fields
def fields_match(node1, node2):
	node1_words = attribute[node1]["fields"]
	node2_words = attribute[node2]["fields"]

	hit = 0 
	for i in range(len(node1_words)):
		for j in range(i+1, len(node2_words)):
			if node1_words[i] == node2_words[j]:
				hit+=1
	return hit


def feature_visualize(train, test, feature):
	data_positive = {}
	data_negative = {}	
	for node1 in tqdm(train.nodes):
		for node2 in train.nodes:
			if (node1, node2) not in train.edges and node1 != node2:

				if feature == "neighbors":
					score = common_neighbors(node1, node2, train)
				if feature == "path":
					score = path(node1, node2)
				if feature == "fields":
					score = fields_match(node1, node2)
				if feature == "keywords":
					score = key_words_match(node1, node2)
				if feature == "keywords_sum":
					score = sum_key_word_count(node1, node2)
				if feature == "neighbors_sum":
					score = sum_neighbors_count(node1, node2)

				if (node1, node2) in test.edges:
					if score in data_positive.keys():
						data_positive[score] +=1
					else:
						data_positive[score] = 1
				else:
					if score in data_negative.keys():
						data_negative[score] +=1
					else:
						data_negative[score] =1

	return data_negative, data_positive

#path_noEdge, path_hasEdge = feature_visualize(train, test, "path")
#nSum_noEdge, nSum_hasEdge = feature_visualize(train, test, "neighbors_sum")
#kSum_noEdge, kSum_hasEdge = feature_visualize(train, test, "keywords_sum")
#field_noEdge, field_hasEdge = feature_visualize(train, test, "fields")
#neighbors_noEdge, neighbors_hasEdge = feature_visualize(train, test, "neighbors")
#keywords_noEdge, keywords_hasEdge = feature_visualize(train, test, "keywords")



fig = plt.figure()

'''
plt.scatter(list(field_hasEdge.keys()), [x / sum(list(field_hasEdge.values())) for x in list(field_hasEdge.values())] , c ="g", label = "positive example")
plt.scatter(list(field_noEdge.keys()), [x / sum(list(field_noEdge.values())) for x in list(field_noEdge.values())] , c= "r", label = "negative exmaple")
plt.legend(loc = "upper left")
plt.xlabel('Number of Fields Matched')
plt.ylabel('Probability')
plt.title('Distribution of Fields Matching Feature')
fig.savefig("fieds.png")

'''

'''
plt.scatter(list(nSum_hasEdge.keys()), [x / sum(list(nSum_hasEdge.values())) for x in list(nSum_hasEdge.values())] , c ="g", label = "positive example")
plt.scatter(list(nSum_noEdge.keys()), [x / sum(list(nSum_noEdge.values())) for x in list(nSum_noEdge.values())] , c= "r", label = "negative exmaple")
plt.legend(loc = "upper left")
plt.xlabel('Sum of Neighbors')
plt.ylabel('Probability')
plt.title('Distribution of Sum of Neighbors Feature')
fig.savefig("neighbors_sum.png")
'''


'''

plt.scatter(list(kSum_hasEdge.keys()), [x / sum(list(kSum_hasEdge.values())) for x in list(kSum_hasEdge.values())] , c ="g", label = "positive example")
plt.scatter(list(kSum_noEdge.keys()), [x / sum(list(kSum_noEdge.values())) for x in list(kSum_noEdge.values())] , c= "r", label = "negative exmaple")
plt.legend(loc = "upper left")
plt.xlabel('Sum of Keywords')
plt.ylabel('Probability')
plt.title('Distribution of Sum of Keywords Feature')
fig.savefig("keywords_sum.png")
'''
'''
ax1 = fig.add_subplot(324)

ax1.scatter(list(field_hasEdge.keys()), [x / sum(list(field_hasEdge.values())) for x in list(field_hasEdge.values())] , c ="g", label = "positive example")
ax1.scatter(list(field_noEdge.keys()), [x / sum(list(field_noEdge.values())) for x in list(field_noEdge.values())] , c= "r", label = "negative exmaple")
plt.legend(loc = "upper left")
plt.xlabel('field')
plt.ylabel('Probability')
plt.title('Distribution of Matching Field Feature')

ax1 = fig.add_subplot(325)

plt.scatter(list(neighbors_hasEdge.keys()), [x / sum(list(neighbors_hasEdge.values())) for x in list(neighbors_hasEdge.values())] , c ="g", label = "positive example")
plt.scatter(list(neighbors_noEdge.keys()), [x / sum(list(neighbors_noEdge.values())) for x in list(neighbors_noEdge.values())] , c= "r", label = "negative exmaple")
plt.legend(loc = "upper left")
plt.legend(loc = "upper left")
plt.xlabel('number of common neighbors')
plt.ylabel('Probability')
plt.title('Distribution of Number of Common Neighbors Feature')
fig.savefig("common_neighbors.png")

ax1 = fig.add_subplot(326)

plt.scatter(list(keywords_hasEdge.keys()), [x / sum(list(keywords_hasEdge.values())) for x in list(keywords_hasEdge.values())] , c ="g", label = "positive example")
plt.scatter(list(keywords_noEdge.keys()), [x / sum(list(keywords_noEdge.values())) for x in list(keywords_noEdge.values())] , c= "r", label = "negative exmaple")
plt.legend(loc = "upper left")
plt.legend(loc = "upper left")
plt.xlabel('Keywords Matching')
plt.ylabel('Probability')
plt.title('Distribution of Number of Matching Keywords Feature')

fig.savefig("keywords.png")

'''





def data_matrix(train, test):
	data = []
	labels = []
	test_data = []
	test_labels = []
	index = 0
	for node1 in tqdm(train.nodes):
		for node2 in train.nodes:
			if (node1, node2) not in train.edges and node1 != node2:
				#calculate path
				path_score = path(node1, node2)

				#neighbors 
				neighbor_score = common_neighbors(node1, node2, train)
				key_score = key_words_match(node1, node2)
				field_score = fields_match(node1, node2)
				kSum = sum_key_word_count(node1, node2)
				nSum = sum_neighbors_count(node1, node2)
				if index <= len(train.nodes)**2.0 - 2*len(train.edges):
					data.append([path_score, neighbor_score, key_score, field_score, kSum, nSum])
					labels.append(int((node1, node2) in test.edges))
				else:
					test_data.append([path_score, neighbor_score, key_score, field_score, kSum, nSum])
					test_labels.append(int((node1, node2) in test.edges))


			index +=1
	return data, labels, test_data, test_labels
'''
data, labels, test_data, test_labels = data_matrix(train, test)

pickle.dump( data, open("data_matrix.p", "wb"))
pickle.dump( labels, open("labels.p", "wb"))
pickle.dump( test_data, open("test_data.p", "wb"))
pickle.dump( test_labels, open("test_labels.p", "wb"))

'''
data = pickle.load(open("data_matrix.p", "rb"))
labels = pickle.load(open("labels.p", "rb"))
test_data= pickle.load(open("test_data.p", "rb"))
test_labels = pickle.load(open("test_labels.p", "rb"))
'''
final = [x for x in labels if x ==1]
final1 = [x for x in test_labels if x ==1]
print (len(labels))
print (len(test_labels))
print (len(final))
print (len(final1))

'''
def model(train, test):

	clf = RandomForestClassifier(max_depth=2, random_state=0)
	#clf = svm.SVC()
	clf.fit(data, labels)
	print(clf.feature_importances_)
	y_pred = []
	for query in test_data:
		y_pred.append(clf.predict([query])[0])



	print (confusion_matrix(test_labels, y_pred))


'''
	positive_n = []
	positive_p = []
	negative_n = []
	negative_p = []
	for [neighbors, path], label in zip(data, labels):
		if label ==1:
			positive_n.append(neighbors)
			positive_p.append(path)
		else:
			negative_n.append(neighbors)
			negative_p.append(path)


	#plt.scatter(positive_p, positive_n, color = "blue")
	plt.scatter(negative_p, negative_n, color = "red")

	plt.show()

'''
model(train, test)






