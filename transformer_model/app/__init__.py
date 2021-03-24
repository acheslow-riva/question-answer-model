from flask import Flask
import pickle
import torch
import torch_neuron
import os
from app.config import config

from elasticsearch_dsl.connections import connections
from haystack import Finder
from haystack.pipeline import ExtractiveQAPipeline
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.reader.farm import FARMReader


def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    host = config[config_name].ELASTIC_HOST
    port = config[config_name].ELASTIC_PORT
    index = config[config_name].QA_INDEX
    es_password = config[config_name].ELASTIC_PASSWORD
    use_traced_model = config[config_name].USE_TRACED_MODEL
    print(index, flush=True)
    doc_store = ElasticsearchDocumentStore(host=host, port=port, username='elastic', password=es_password, index=index)
    es = connections.create_connection(hosts=[config[config_name].ELASTIC_URL], 
        http_auth=(config[config_name].ELASTIC_USER, es_password)
    )
    app.es = es
    retriever = ElasticsearchRetriever(document_store=doc_store)
    model_name = '/transformer_model/app/static/data/language_model/roberta-base-squad2'
    saved_locally = os.path.exists(model_name)
    if not saved_locally:
        model_name = "deepset/roberta-base-squad2"
    reader = FARMReader(model_name_or_path=model_name, num_processes=0, use_gpu=False)
    if not saved_locally:
        reader.save("/transformer_model/app/static/data/language_model/roberta-base-squad2") 
    app.finder = ExtractiveQAPipeline(reader, retriever)
    app.finder.get_node("Reader").inferencer.batch_size=1

    if use_traced_model:
        direct = os.listdir('app/static/data/language_model')
        if not 'traced_model.pt' in direct:
            app.logger.info("Model not found. Compiling.")
            app.logger.info("Compiling model.")
            inputs = pickle.load(open('app/static/single.p', 'rb'))
            app.finder.get_node("Reader").inferencer.model.language_model.model.eval()
            app.finder.get_node("Reader").inferencer.model.language_model.model.config.return_dict = False
            app.finder.get_node("Reader").inferencer.model.language_model.model.encoder.config.output_hidden_states = False
            model = torch.neuron.trace(app.finder.get_node("Reader").inferencer.model.language_model.model, example_inputs=inputs)
            model.save('app/static/data/language_model/traced_model.pt')
            app.model = model
        else:
            app.logger.info("Model exists. Loading")
            model = torch.jit.load('app/static/data/language_model/traced_model.pt')
        app.finder.get_node("Reader").inferencer.model.language_model.model = model
        app.finder.get_node("Reader").inferencer.model.language_model.model.eval()
        query = "What does AHRQ stand for?"
        try:
            res = app.finder.run(query, top_k_retriever=10, top_k_reader=1)
            response = res['answers'][0]['answer']
            if response != "Agency for Healthcare Research and Quality":
                app.logger.info("Model error. Re-compiling.")
                app.finder = ExtractiveQAPipeline(reader, retriever)
                app.logger.info("Compiling model.")
                inputs = pickle.load(open('app/static/single.p', 'rb'))
                app.finder.get_node("Reader").inferencer.model.language_model.model.eval()
                app.finder.get_node("Reader").inferencer.model.language_model.model.config.return_dict = False
                app.finder.get_node("Reader").inferencer.model.language_model.model.encoder.config.output_hidden_states = False
                model = torch.neuron.trace(app.finder.get_node("Reader").inferencer.model.language_model.model, example_inputs=inputs)
                model.save('app/static/data/language_model/traced_model.pt')
                app.model = model
            else:
                app.logger.info("Primed model")
        except RuntimeError:
            app.logger.error("Model not working properly")
    from app.main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app
