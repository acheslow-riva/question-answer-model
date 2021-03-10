import os
import pickle
import torch
import torch_neuron
from time import time

from haystack.reader.farm import FARMReader
from flask import request, current_app, jsonify, abort, url_for
# from transformers.modeling_roberta import RobertaModel
from . import main

# from elasticsearch import Elasticsearch

@main.route('/')
def index():
    return jsonify({"hello":"world"})

@main.route('/load_default_model')
def default_model():
    model_name = "deepset/roberta-base-squad2"
    current_app.finder.reader = FARMReader(model_name_or_path=model_name, num_processes=0, use_gpu=False)
    current_app.finder.reader.inferencer.batch_size=1
    return jsonify({"default model": 'loaded'})

@main.route('/load_traced_model')
def load_traced_model():
    direct = os.listdir('app/static/data/language_model')
    if not 'traced_model.pt' in direct:
        inputs = pickle.load(open('app/static/single.p', 'rb'))
        current_app.finder.reader.inferencer.model.language_model.model.eval()
        traced_model = torch.neuron.trace(current_app.finder.reader.inferencer.model.language_model.model, example_inputs=inputs) 
        traced_model.save('app/static/data/language_model/traced_model.pt')
    current_app.finder.reader.inferencer.model.language_model.model = torch.jit.load('app/static/data/language_model/traced_model.pt')
    current_app.finder.reader.inferencer.model.language_model.model.eval()
    return jsonify({'done':'loading'})

@main.route('/get_answers')
def get_answers(query=None):
    start = time()
    query = request.args.get('query')
    top_k_retriever = request.args.get('top_k_retriever', 10)
    if not query: query = 'what does ahrq stand for'
    response = current_app.finder.get_answers(query, top_k_retriever=top_k_retriever, top_k_reader=1)
    return jsonify({'response': response,
                    'Elapsed time': time()-start})
