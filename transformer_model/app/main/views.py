from time import time

from elasticsearch_dsl import Q, Search
from flask import request, current_app, jsonify, abort
from haystack import Document

from . import main


@main.route('/')
def index():
    return jsonify({"hello":"question-answer"})

@main.route('/get_answers', methods=["GET", "POST"])
@main.route('/ask', methods=["GET", "POST"])
def ask():
    start = time()
    args = request.args or {}
    json = request.json or {}
    query = args.get('query', args.get('q', json.get("query", json.get("q"))))
    top_k_retriever = args.get('top_k_retriever', json.get('top_k_retriever', 10))
    top_k = int(args.get("rows", args.get("top_k", json.get("rows", json.get("top_k", 1)))))
    body = json.get("body")
    url = args.get('url')
    if not query:
        abort(400, "no query sent in the q or query url parameter")
    if url:
        q = Q("term", url=url)
        s = Search(index=current_app.config.get('QA_INDEX'))
        docs = [Document.from_dict(hit.to_dict()) for hit in s.query(q).scan()]
        if not docs:
            abort(404, f"document {url} not found")
        response = current_app.finder.get_node("Reader").predict(query=query, documents=docs, top_k=top_k)
    elif body:
        doc = Document(text=body)
        response = current_app.finder.get_node("Reader").predict(query=query, documents=[doc], top_k=top_k)
    else:
        response = current_app.finder.run(query, top_k_retriever=top_k_retriever, top_k_reader=top_k)
    return jsonify({'response': response,
                    'Elapsed time': time() - start})