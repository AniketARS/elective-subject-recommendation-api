from flask import Flask, request, jsonify
from flask_cors import CORS
from annoy import AnnoyIndex
from functools import reduce
import pickle
import os

app = Flask(__name__)

# Enable CORS for API endpoints
cors = CORS(app, resources={r'/api/*': {'origins': '*'}})

# Variables
f = 9
K_MAX = 5
ANNOY = AnnoyIndex(f, 'angular')
SUB_TO_IDX = {}
IDX_TO_SUB = {}


@app.route("/api/v1/similar-subject", methods=['POST'])
def similar_subject():
    '''
        This method will accept API request from client for making recommendation for subject
    '''
    args = parse_args(request.get_json())
    if 'error' in args.keys():
        return jsonify(args)
    recommendations = make_recommendations(args)
    return recommendations


@app.route("/api/v1/subject-list", methods=['GET'])
def subject_list():
    subjects = SUB_TO_IDX.keys()
    return jsonify({'subjects': list(subjects)})


def parse_args(data):
    subject = data['subject']
    # loading index for subject
    try:
        idx = SUB_TO_IDX[subject]
        k = data['k'] if 'k' in data.keys() else 3
    except KeyError:
        return {"error": "Sorry given subject is not in database"}
    return {'id': idx, 'k': min(k, K_MAX)}


def make_recommendations(args):
    recommended_idx = ANNOY.get_nns_by_item(args['id'], args['k']+1, include_distances=True)
    res = []
    for id_, distance in zip(recommended_idx[0][1:], recommended_idx[1][1:]):
        d = {
            'name': IDX_TO_SUB[id_],
            'score': 1-distance
        }
        res.append(d)
    return jsonify(res)

def load_variables():
    '''
        This function loads all the variable required to make recommendations for the given
        subject through API
    '''
    global SUB_TO_IDX, IDX_TO_SUB
    # Common Path to all variables
    mid_path = [os.curdir, 'model', 'KNB_MODEL', 'ONE_HOT']

    print("LOADING ANNOY...")
    ANNOY.load(os.path.join('', *(mid_path + ['tree.ann'])))

    print("LOADING SUBJECT-TO-INDEX MAP...")
    with open(os.path.join('', *(mid_path + ['subject2idx.pkl'])), 'rb') as f:
        SUB_TO_IDX = pickle.load(f)

    print("LOADING INDEX-TO-SUBJECT MAP...")
    with open(os.path.join('', *(mid_path + ['idx2subject.pkl'])), 'rb') as f:
        IDX_TO_SUB = pickle.load(f)

load_variables()

if __name__ == '__main__':
    app.run()