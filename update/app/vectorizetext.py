import logging
import os

from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.dense import DensePassageRetriever


logging.info("Connecting to elasticsearch")

document_store = ElasticsearchDocumentStore(
    host=os.environ.get("ELASTIC_HOST"),
    username=os.environ.get("ELASTIC_USER"), 
    password=os.environ.get("ELASTIC_PASSWORD"),
    index=os.environ.get("QA_INDEX")
)

retriever = DensePassageRetriever(document_store=document_store,
                query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                max_seq_len_query=64,
                max_seq_len_passage=256,
                batch_size=16,
                use_gpu=False,
                embed_title=False,
                use_fast_tokenizers=True)

logging.info("Updating stuff")
document_store.update_embeddings(retriever)