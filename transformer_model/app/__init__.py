from flask import Flask
from app.config import config

from haystack import Finder
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.reader.farm import FARMReader
from haystack.utils import print_answers

def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    host = config[config_name].ELASTIC_URL
    port = config[config_name].ELASTIC_PORT
    index = config[config_name].ELASTIC_INDEX

    doc_store = ElasticsearchDocumentStore(host=host, port=port, username='elastic', password='changeme', index=index)

    retriever = ElasticsearchRetriever(document_store=doc_store)
    model_name = "deepset/roberta-base-squad2"
    reader = FARMReader(model_name_or_path=model_name, num_processes=0, use_gpu=False)
    app.finder = Finder(reader, retriever)
    app.finder.reader.inferencer.batch_size=1

    from app.main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app