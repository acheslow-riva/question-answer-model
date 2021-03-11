from flask import Flask
import pickle
import torch
import torch_neuron
import os
from app.config import config

from haystack import Finder
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.reader.farm import FARMReader

def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    host = config[config_name].ELASTIC_URL
    port = config[config_name].ELASTIC_PORT
    index = config[config_name].ELASTIC_INDEX
    es_password = config[config_name].ELASTIC_PASSWORD
    use_traced_model = config[config_name].USE_TRACED_MODEL
    print(index, flush=True)
    doc_store = ElasticsearchDocumentStore(host=host, port=port, username='elastic', password=es_password, index=index)

    retriever = ElasticsearchRetriever(document_store=doc_store)
    model_name = '/transformer_model/app/static/data/language_model/roberta-base-squad2'
    if not os.path.exists(model_name):
        model_name = "deepset/roberta-base-squad2"
    reader = FARMReader(model_name_or_path=model_name, num_processes=0, use_gpu=False)
    app.finder = Finder(reader, retriever)
    app.finder.reader.inferencer.batch_size=1

    if use_traced_model:
        direct = os.listdir('app/static/data/language_model')
        if not 'traced_model.pt' in direct:
            app.logger.info("Model not found. Compiling.")
            inputs = pickle.load(open('app/static/single.p', 'rb'))
            app.finder.reader.inferencer.model.language_model.model.eval()
            app.finder.reader.inferencer.model.language_model.model.config.return_dict = False
            model = torch.neuron.trace(app.finder.reader.inferencer.model.language_model.model, example_inputs=inputs) 
            model.save('app/static/data/language_model/traced_model.pt')
        else:
            app.logger.info("Model exists. Loading")
            model = torch.jit.load('app/static/data/language_model/traced_model.pt')
        app.finder.reader.inferencer.model.language_model.model = model
        app.finder.reader.inferencer.model.language_model.model.eval()

    from app.main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app
