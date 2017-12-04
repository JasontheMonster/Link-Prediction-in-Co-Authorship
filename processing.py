"""
Grabs data from MAG and processes it
"""

import json
from unidecode import unidecode
import pickle
import requests
import os
from collections import defaultdict

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
    url_list = build_url()
    headers = {
        # "Host": "westus.api.cognitive.microsoft.com",
        "Ocp-Apim-Subscription-Key": "e728f81ce2c443e6b17d3da63b469b22"
    }
    path = "./mag_raw/"
    for i, url in enumerate(url_list):
        r = requests.get(url, headers=headers)
        data = r.json()
        file_name = path + str(i) + "_author_attributes.json"
        with open(file_name, 'w') as f:
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


def build_tables():
    path = "./mag_raw/"
    json_files = [path + pos_json for pos_json in os.listdir(path)]
    db = defaultdict(lambda: defaultdict(list))
    for fl in json_files:
        with open(fl, 'r') as f:
            try:
                data = json.load(f)['entities']
            except KeyError:
                pass
            else:
                for entry in data:
                    keywords = entry['W']
                    title = entry["Ti"]
                    try:
                        fields = [f["FN"] for f in entry["F"]]
                        authors = [a["AuN"] for a in entry["AA"]]
                    except KeyError:
                        pass
                    else:
                        for a in authors:
                            db[a]["fields"].extend(fields)
                            db[a]["keywords"].extend(keywords)
                            db[a]["papers"].append(title)
                            fields = list(set(db[a]["fields"]))
                            kw = list(set(db[a]["keywords"]))
                            db[a]["fields"] = fields
                            db[a]["keywords"] = kw
    with open('attributes.p', 'w') as fp:
        json.dump(db, fp)

build_tables()