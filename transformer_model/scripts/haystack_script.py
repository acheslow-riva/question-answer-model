from haystack import Finder
from haystack.preprocessor.cleaning import clean_wiki_text
# from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers

# Connect to a locally running instance of Elasticsearch

from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
# document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="ahrq", search_fields='body')
document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="ahrq")

from haystack.retriever.sparse import ElasticsearchRetriever
retriever = ElasticsearchRetriever(document_store=document_store)

# Alternative: An in-memory TfidfRetriever based on Pandas dataframes for building quick-prototypes with SQLite document store.

# from haystack.retriever.sparse import TfidfRetriever
# retriever = TfidfRetriever(document_store=document_store)


# Load a  local model or any of the QA models on
# Hugging Face's model hub (https://huggingface.co/models)

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", num_processes=0, use_gpu=False)

finder = Finder(reader, retriever)

question = "What department is AHRQ a part of?"
prediction = finder.get_answers(question, top_k_retriever=10, top_k_reader=5)

print_answers(prediction, details="medium")