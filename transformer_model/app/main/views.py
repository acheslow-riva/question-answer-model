import os
import pickle
import torch
import torch_neuron
from time import time

from elasticsearch_dsl import Q, Search
from haystack import Document
from haystack.reader.farm import FARMReader
from flask import request, current_app, jsonify, abort, url_for
# from transformers.modeling_roberta import RobertaModel
from . import main

# from elasticsearch import Elasticsearch

@main.route('/')
def index():
    return jsonify({"hello":"question-answer"})

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
def get_answers():
    start = time()
    query = request.args.get('query', request.args.get('q'))
    top_k_retriever = request.args.get('top_k_retriever', 10)
    top_k = int(request.args.get("rows", 1))
    if not query: query = 'what does ahrq stand for'
    response = current_app.finder.get_answers(query, top_k_retriever=top_k_retriever, top_k_reader=top_k)
    return jsonify({'response': response,
                    'Elapsed time': time()-start})

@main.route('/ask')
def ask():
    start = time()
    query = request.args.get('query', request.args.get('q'))
    url = request.args.get('url')
    if not url or not query:
        abort(401)
    top_k = int(request.args.get("rows", 1))
    q = Q("term", url=url)
    s = Search(index=current_app.config.get('QA_INDEX'))
    docs = [Document.from_dict(hit.to_dict()) for hit in s.query(q).scan()]
    response = current_app.finder.reader.predict(query=query, documents=docs, top_k=top_k)
    return jsonify({'response': response,
                    'Elapsed time': time()-start})