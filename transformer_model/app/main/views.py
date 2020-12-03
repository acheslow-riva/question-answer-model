import os
import pickle
import torch
import torch_neuron
from time import time

from flask import request, current_app, jsonify, abort, url_for
from transformers.modeling_roberta import RobertaModel
from . import main

# from elasticsearch import Elasticsearch

@main.route('/')
def index():
     
    # print(f'Count: {count}', flush=True)

    # return jsonify({"count": count})
    return jsonify({"hello":"Ryan"})

@main.route('/model_info')
def model_info():
    model_head = current_app.finder.reader.inferencer.model.prediction_heads
    return jsonify({"prediction_heads": dir(model_head[0])})

@main.route('/load')
def load():
    direct = os.listdir('app/static/data/language_model')
    if not 'traced_model.pt' in direct:
        inputs = pickle.load(open('app/static/single.p', 'rb'))
        # inputs = (batch['input_ids'], batch['padding_mask'], batch['segment_ids'])
        current_app.finder.reader.inferencer.model.language_model.model.eval()
        # TODO: Figure out if example model inputs batch size affects model execution time.
        # traced_model = torch.jit.trace(current_app.finder.reader.inferencer.model.language_model.model, inputs) # Trace with AWS Neuron instead of pytorch
        # torch.jit.save(traced_model, 'app/static/data/language_model/traced_model.pt')
        traced_model = torch.neuron.trace(current_app.finder.reader.inferencer.model.language_model.model, example_inputs=inputs) 
        traced_model.save('app/static/data/language_model/traced_model.pt')
    current_app.finder.reader.inferencer.model.language_model.model = torch.jit.load('app/static/data/language_model/traced_model.pt')
    current_app.finder.reader.inferencer.model.language_model.model.eval()
    return jsonify({'done':'loading'})

@main.route('/get_answers')
def get_answers(query=None):
    start = time()
    query = request.args.get('query')
    if not query: query = 'what does ahrq stand for'
    response = current_app.finder.get_answers(query, top_k_retriever=5, top_k_reader=1)
    return jsonify({'doc 1 text': response,
                    'Elapsed time': time()-start})
