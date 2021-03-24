from time import time

from elasticsearch_dsl import Q, Search
from flask import request, current_app, jsonify, abort
from haystack import Document

from . import main


@main.route('/')
def index():
    return jsonify({"hello":"question-answer"})

@main.route('/get_answers')
@main.route('/ask')
def ask():
    start = time()
    query = request.args.get('query', request.args.get('q'))
    top_k_retriever = request.args.get('top_k_retriever', 10)
    top_k = int(request.args.get("rows", request.args.get("top_k", 1)))
    url = request.args.get('url')
    if not query:
        abort(400)
    if url:
        q = Q("term", url=url)
        s = Search(index=current_app.config.get('QA_INDEX'))
        docs = [Document.from_dict(hit.to_dict()) for hit in s.query(q).scan()]
        response = current_app.finder.get_node("Reader").predict(query=query, documents=docs, top_k=top_k)
    else:
        response = current_app.finder.run(query, top_k_retriever=top_k_retriever, top_k_reader=top_k)
    return jsonify({'response': response,
                    'Elapsed time': time() - start})