import os
import pickle
import torch
import torch_neuron
from time import time

from haystack.reader.farm import FARMReader
from flask import request, current_app, jsonify, abort, url_for
from transformers.modeling_roberta import RobertaModel
from . import main

# from elasticsearch import Elasticsearch

@main.route('/')
def index():
     
    # print(f'Count: {count}', flush=True)

    # return jsonify({"count": count})
    return jsonify({"hello":"Ryan"})

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

@main.route('/load_torch_model')
def load_torch():
    current_app.finder.reader.inferencer.model.language_model.model = torch.jit.load('app/static/data/language_model/torch_traced_model.pt')
    current_app.finder.reader.inferencer.model.language_model.model.eval()
    return jsonify({'done':'loading'})

@main.route('/load_neuron_model')
def load_neuron():
    current_app.finder.reader.inferencer.model.language_model.model = torch.jit.load('app/static/data/language_model/neuron_traced_model.pt')
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
