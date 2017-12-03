#!/usr/bin/python
## wget -N http://dblp.uni-trier.de/xml/dblp.xml.gz
## then run this script
import json
import networkx as nx
import time
import pickle
 
with open("tmp_dblp_coauthorship.json") as f:
    d = json.load(f)


#training set
training = []
#testing set
testing = []
#build a undirected unweighted graph with nodes as author and edge as if they have been coauthor
for author1, author2, year in d:
  #training
  if year >= 1990 and year <2000:
    training.append((author1.encode('utf-8'), author2.encode('utf-8')))
  #testing set
  elif year >= 2000 and year <= 2004:
    testing.append((author1.encode('utf-8'), author2.encode('utf-8')))

#create two networkx graph and add edge with list training and test

train = nx.Graph()
test = nx.Graph()

train.add_edges_from(training)
test.add_edges_from(testing)

pickle.dump(train, open("graphs/train.p", "wb"), protocol = 0)
pickle.dump(test, open("graphs/test.p", "wb"), protocol = 0)





#
