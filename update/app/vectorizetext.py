import logging
import os

from elasticsearch_dsl.connections import connections
from elasticsearch_dsl import Q, Search
from haystack import Document
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.dense import DensePassageRetriever


logging.info("Connecting to elasticsearch")
es = connections.create_connection(hosts=[os.environ.get("ELASTIC_URL")],
    http_auth=(os.environ.get("ELASTIC_USER"), os.environ.get("ELASTIC_PASSWORD"))
)
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

s = Search(index=os.environ.get("QA_INDEX"))
total = s.count()
batch_size = 100
total_batches = total // batch_size + (0 if total % batch_size == 0 else 1)
counter = 0
current_batch = 1
batch_docs = []
delete_ids = []
for hit in s.params(scroll="1d").scan():
    if counter == batch_size:
        batch_emb = retriever.embed_passages(batch_docs)
        for emb, doc in zip(batch_emb, batch_docs):
            doc.embedding = emb
        logging.info(f"Updating batch {current_batch} of {total_batches}")
        document_store.write_documents(batch_docs)
        s.query(Q("ids", values=delete_ids)).delete()
        delete_ids = []
        batch_docs = []
        counter = 0
        current_batch += 1
        continue
    doc = Document.from_dict(hit.to_dict())
    delete_ids.append(hit.meta.id)
    batch_docs.append(doc)
    counter += 1

# Last batch
if batch_docs:
    batch_emb = retriever.embed_passages(batch_docs)
    for emb, doc in zip(batch_emb, batch_docs):
        doc.embedding = emb
    document_store.write_documents(batch_docs)
    s.query(Q("ids", values=delete_ids)).delete()
# https://github.com/deepset-ai/haystack/issues/601#issuecomment-729469535
# document_store.update_embeddings(retriever)