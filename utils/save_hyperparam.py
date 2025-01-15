import json

def save_hyperparam(dict):
    with open('hyperparam.json', 'w') as f:
        json.dump(dict, f)