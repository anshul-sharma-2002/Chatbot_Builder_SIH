# from crypt import methods
# from email import message
from sys import flags
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chat import get_response
import pymongo
import pandas as pd
import spacy
# from spacy import displacy
from sentence_transformers import SentenceTransformer, util
from random import random
import random
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

app = Flask(__name__)
CORS(app)
nlp = spacy.load('en_core_web_md')
client = pymongo.MongoClient('localhost', 27017)
db = client.get_database('sih')
# records = db.chats

def ner(rawtext, entity):
    doc = nlp(rawtext)
    d = []
    ans = []
    ORG_named_entity = pd.Series()
    PERSON_named_entity = pd.Series()
    GPE_named_entity = pd.Series()
    for ent in doc.ents:
        d.append((ent.label_, ent.text))
        df = pd.DataFrame(d, columns=('named entity', 'output'))
        ORG_named_entity = df.loc[df['named entity'] == 'ORG']['output']
        PERSON_named_entity = df.loc[df['named entity'] == 'PERSON']['output']
        GPE_named_entity = df.loc[df['named entity'] == 'GPE']['output']
        # MONEY_named_entity = df.loc[df['named entity'] == 'MONEY']['output']
    if entity == "Organisation":
        ans = [x for _, x in ORG_named_entity.items()]
    elif entity == "Name":
        ans = [x for _, x in PERSON_named_entity.items()]
    elif entity == "Place":
        ans = [x for _, x in GPE_named_entity.items()]
    # monLst = [x for _, x in MONEY_named_entity.items()]
    if len(ans) == 0:
        return False

    return True

# @app.route("/", methods=["GET"])
# def index_get():
#     return render_template("base.html")

curID = "1"

def simAnalysis(k, text):
    nodeTb = db.node_info
    node = nodeTb.find_one({'id':k})
    l = node["pattern"]
    mxScr = 0
    for msg in l:
        emb1 = model.encode(msg, convert_to_tensor=True)
        emb2 = model.encode(text, convert_to_tensor=True)
        scr = util.pytorch_cos_sim(emb1, emb2)
        mxScr = max(mxScr, scr)

    return mxScr
    
# cbID = random.randInt(1000,9999)
idD = db.tree.find_one()
print(idD)
tmpD = idD
def flow(k, text, d):
    nxtId = k
    
    if k==None:
        return None, None, None
    # print(node)
    for i in d[k].keys():
        nxtId = i
    nodeTb = db.node_info
    node = nodeTb.find_one({'id':nxtId})
    resp = node["responses"]

    if node["rb"] == 2:
        entity = resp["ner"]
        flag = ner(text, entity)
        response=str(flag)
        if flag:
            response = resp["prompt"]

        return response, nxtId, d[k]
    else:
        nxtId=k
        mxScore = 0
        th = 0.5
        # print
        if len(d[k].keys()) == 0:
            return None, None, None
        for i in d[k].keys():
            curScore = simAnalysis(i, text)
            if curScore > mxScore:
                mxScore = curScore
                nxtId = i
        print(f"\nSimilarity Score: {mxScore}\n")
        if mxScore >= th:
            nodeTb = db.node_info
            node = nodeTb.find_one({'id':nxtId})
            resp = node["responses"]
            if node['rb'] == 0:#text
                response = random.choice(resp["text"])
            elif node['rb'] == 1:#option
                response = resp["prompt"]
                for i in resp["option"]:
                    response += "$" + i
                
            return response, nxtId, d[k]
    print(k,d)
    return None, k, d

@app.route("/predict", methods=["POST"])
# @app.post("/predict")
def predict():
    global curID
    global tmpD
    text = request.get_json().get("message")
    response, curID, tmpD = flow(curID, text, tmpD)
    flag=0
    if response == None:
        print("\nEntered no flow\n")
        response = get_response(text)
        flag=1
    if(flag==1):
        flag=0
        print("\nback to flow\n")
    if curID == None:
        curID = "1"
    message = {"answer": response}
    # entity = ner(text)
    # user_input = {'question': text, 'answer': response}
    # records.insert_one(user_input)
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)