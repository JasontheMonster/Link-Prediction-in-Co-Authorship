"""
Grabs data from MAG and processes it
"""

import json
from unidecode import unidecode
import pickle
import requests
import itertools

write = True
read = True


def generate_authors(read, write):
    """
    generate a list of authors, for strings formatted to remove accents and special characters
    """
    if write:
        with open('dblp_coauthorship.json', 'r') as f:
            d = json.load(f)

        authors = set()
        for author1, author2, year in d:
            author1 = unidecode(author1)
            author2 = unidecode(author2)
            authors.update([author1.replace('.', '').replace('-', '').lower(),
                            author2.replace('.', '').replace('-', '').lower()])

        with open('authors', 'wb') as fp:
            pickle.dump(list(authors), fp)

    if read:
        with open('authors', 'rb') as fp:
            item_list = pickle.load(fp)
            return item_list


def get_json():
    url = "https://westus.api.cognitive.microsoft.com/academic/v1.0/evaluate?expr=" \
          "Composite(F.FN=='fluid dynamics')&model=latest&count=10000&offset=0&attributes=Y,Ti,Id,F.FN,AA.AuN,W"
    headers = {
        "Ocp-Apim-Subscription-Key": ""  # add own key
    }
    path = "./database/"
    # for i, url in enumerate(url_list):
    r = requests.get(url, headers=headers)
    data = r.json()
    # file_name = path + str(i) + "_author_attributes.json"
    file_name = "fluid_dynamics.json"
    with open(path+file_name, 'w') as f:
        json.dump(data, f)


def build_url():
    with open('authors', 'rb') as fp:
        author_list = pickle.load(fp)
    url_list = []
    for i in range(0, len(author_list), 50):
        s = []
        for author in author_list[i:i+50]:
            a = "AA.AuN=='%s'" % author
            s.append(a)
        s1 = ",".join(s)
        s2 = "Composite(OR(%s))" % s1
        s3 = "https://westus.api.cognitive.microsoft.com/academic/v1.0/evaluate?expr=%s&model=latest" \
               "&count=1000&offset=0&attributes=W,Ti,AA.AuN,F.FN,Y HTTP/1.1" % s2
        url_list.append(s3)
    return url_list


def create_edge_list():
    with open('./database/fluid_dynamics.json', 'rb') as fp:
        db = json.load(fp)['entities']
        edge_list = set()
        authors = set()
        for paper in db:
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



