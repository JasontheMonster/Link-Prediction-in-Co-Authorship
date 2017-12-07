"""
Grabs data from Microsoft Academic Graph API and data processing
"""

import json
import pickle
import requests
import itertools
from collections import defaultdict
from time import sleep


def get_json():
    """
    send request to Microsoft Academic Graph for information on papers on fluid dynamics
    generates a json file
    """
    url1 = "https://westus.api.cognitive.microsoft.com/academic/v1.0/evaluate?expr=" \
          "Composite(F.FN=='fluid dynamics')&model=latest&count=10000&offset=0&attributes=Y,Ti,Id,F.FN,AA.AuN,W"
    headers = {
        "Ocp-Apim-Subscription-Key": ""  # add own key
    }
    path = "./database/"
    r = requests.get(url1, headers=headers)
    data = r.json()
    file_name = "fluid_dynamics.json"
    with open(path+file_name, 'w') as f:
        json.dump(data, f)
    url2 = "https://westus.api.cognitive.microsoft.com/academic/v1.0/evaluate?expr=" \
           "Composite(F.FN=='fluid dynamics')&model=latest&count=10000&offset=1000&attributes=Y,Ti,Id,F.FN,AA.AuN,W"
    headers = {
        "Ocp-Apim-Subscription-Key": ""  # add own key
    }
    r = requests.get(url2, headers=headers)
    data = r.json()
    file_name = "fluid_dynamics2.json"
    with open(path + file_name, 'w') as f:
        json.dump(data, f)


def create_edge_list():
    """
    creates and pickles the edges list and vertex set
    """
    edge_list = set()
    authors = set()
    with open('./database/fluid_dynamics.json', 'rb') as fp:
        db = json.load(fp)['entities']
        for paper in db:
            author_list = [a["AuN"] for a in paper["AA"]]
            edges = list(itertools.combinations(author_list, 2))
            year = [paper['Y']]*len(edges)
            edge_year = list(zip(edges, year))
            authors.update(author_list)
            edge_list.update(edge_year)
    with open('./database/fluid_dynamics2.json', 'rb') as fp:
        db2 = json.load(fp)['entities']
        for paper in db2:
            author_list = [a["AuN"] for a in paper["AA"]]
            edges = list(itertools.combinations(author_list, 2))
            year = [paper['Y']]*len(edges)
            edge_year = list(zip(edges, year))
            authors.update(author_list)
            edge_list.update(edge_year)

    with open("database/edges.p", 'wb') as f:
        pickle.dump(edge_list, f)
    with open("database/authors.p", 'wb') as f:
        pickle.dump(authors, f)


# Ti,AA.AuN,W,F.FN
def node_attributes(authorlist=None):
    """
    for every author in the author list, gets information on the number of collaborations, list of keywords,
    list of fields he/she has worked in, and number of papers published
    creates a json file to store data
    """
    if authorlist is None:
        with open("./database/train_authors.p", 'rb') as f:
            authors = pickle.load(f)
    else:
        authors = authorlist

    with open('./database/attributes.json', 'r') as f:
        attributes = json.load(f)
    counter = 0
    for i, author in enumerate(authors):
        if author in attributes.keys():
            print(str(author) + " was skipped because info was already found")
            continue
        counter += 1
        url = "https://westus.api.cognitive.microsoft.com/academic/v1.0/calchistogram?expr=Composite(AA.AuN=='%s')" \
              "&model=latest&attributes=Ti,AA.AuN,W,F.FN&count=100&offset=0" % author
        headers = {
            "Ocp-Apim-Subscription-Key": ""  # add own key
        }
        r = requests.get(url, headers=headers)
        print("round " + str(i) + " r: " + str(r))
        d = r.json()
        data = d['histograms']
        attributes[author] = dict()
        for entry in data:
            if entry["attribute"] == "Ti":
                num_papers = entry["distinct_values"]
                attributes[author]["num_papers"] = num_papers
            elif entry["attribute"] == "AA.AuN":
                num_collaborations = entry["distinct_values"]
                attributes[author]["num_collaborations"] = num_collaborations
            elif entry["attribute"] == "W":
                keywords = list(set([e["value"] for e in entry["histogram"]]))
                attributes[author]["keywords"] = keywords
            else:
                assert entry["attribute"] == "F.FN", "attribute type is not field"
                fields = list(set([e["value"] for e in entry["histogram"]]))
                attributes[author]["fields"] = fields
        if counter == 20:
            print("writing to file")
            counter = 0
            with open("database/attributes.json", 'w') as f:
                json.dump(attributes, f)
        sleep(10)

    with open("database/attributes.json", 'w') as f:
        json.dump(attributes, f)


def get_edge_weights():
    """
    for each edge, the weight is the number of collaborations done between the two authors
    """
    with open("database/edges.p", 'rb') as f1:
        edges = pickle.load(f1)

    edge1 = [x[0] for x in edges]
    weights = defaultdict(lambda: 0)
    for edge in edge1:
        weights[edge] += 1

    w = dict(weights)

    with open("database/weights.p", 'wb') as f:
        pickle.dump(w, f)

