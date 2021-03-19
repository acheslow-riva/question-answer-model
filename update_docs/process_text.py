from datetime import datetime
import os
import re
from urllib import parse

from elasticsearch_dsl.connections import connections
from elasticsearch_dsl import A, Q, Search
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.preprocessor.preprocessor import PreProcessor

"""
The first Basic_QA_Pipeline.ipynb went through the ahrq index (an index containing the text of www.ahrq.gov html sites),
and saved them to text files locally. Then, it called `haystack.preprocessor.utils.convert_files_to_dicts`, passing it
the clean_url_text function defined below. To replicate this, I modified the crawler to capture the text correctly so
the paragraph splitting works. The convert_files_to_dicts function calls the `haystack.file_converter.txt.convert`
function. So, the order of the text processing is convert which just reads in the webpage text, calls clean_url_text(text)
then does the paragraph splitting on '\n\n', and returns the list of split paragraphs (documents). These documents
should have a `text` field key whose value is the paragraph, and a `meta` field whose value is another object with the
key `url` whose value is the page's url.
"""

INDEX = "ahrq_qa2"

def farm_convert_text(text):
    pages = text.split("\f")
    cleaned_pages = []
    for page in pages:
        lines = page.splitlines()
        cleaned_lines = []
        for line in lines:
            words = line.split()
            digits = [word for word in words if any(i.isdigit() for i in word)]
            # remove lines having > 40% of words as digits AND not ending with a period(.)
            if words and len(digits) / len(words) > 0.4 and not line.strip().endswith("."):
                continue
            cleaned_lines.append(line)
        page = "\n".join(cleaned_lines)
        cleaned_pages.append(page)
    text = "".join(cleaned_pages)
    return text



def clean_url_text(text):
    # get rid of multiple new lines
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    # remove extremely short lines, combine small paragraphs into larger ones
    lines = text.split("\n")
    cleaned = []
    multi_lines = ''
    for l in lines:
        if len(l) > 100:
            multi_lines += l + '\n\t'
        if len(l) > 500:
            cleaned.append(multi_lines)
            multi_lines = ''
    if multi_lines: cleaned.append(multi_lines) 
    text = "\n\n".join(cleaned)
    return text


def text_to_docs(text, url):
    documents = []
    text = clean_url_text(text)
    for para in text.split("\n\n"):
        if not para.strip():  # skip empty paragraphs
            continue
        documents.append(
            {
                "text": para, 
                "meta": {
                    "url": url,
                    "@timestamp": datetime.utcnow(),
                }
            }
        )
    return documents

def get_hostname(url):
    parsed = parse.urlparse(url)
    if parsed.port:
        return parsed.netloc.replace(f':{parsed.port}', '')
    else:
        return parsed.netloc


es = connections.create_connection(
    hosts=[os.environ.get("ELASTIC_URL")],
    http_auth=("elastic", os.environ.get("ELASTIC_PASSWORD"))
)

document_store = ElasticsearchDocumentStore(
    host=get_hostname(os.environ.get("ELASTIC_URL")),
    username="elastic", 
    password=os.environ.get("ELASTIC_PASSWORD"),
    index=INDEX
)
processor = PreProcessor(clean_empty_lines=True,
                         clean_whitespace=True,
                         clean_header_footer=True,
                         split_by="word",
                        #  split_length=200,
                         split_respect_sentence_boundary=True)
filters = [
    Q("term", siteDomain="www.ahrq.gov"), # just the www.ahrq.gov pages
    Q("term", mimeType__keyword="text/html"), # just the html page
    Q("exists", field="parsedContent"), # make sure there is text on page
    Q("exists", field="detectedDate"), # use this to 
    Q("range", tstamp={"gte": "2021-02-01"}),
]
if es.indices.exists(INDEX):
    # Check lastModified or detectedDate
    s = Search(index=INDEX)
    s.aggs.bucket("max_timestamp", A("max", field="@timestamp"))
    max_date = s.extra(size=0).execute().aggs.max_timestamp.value
    filters.append(
        Q("range", **{"@timestamp": {"gte": max_date}})
    )

q = Q("bool", filter=filters)
extra = {"_source": "parsedContent"}
s = Search(index="webpages").extra(**extra)
all_docs = []
for hit in s.query(q).scan():
    docs = text_to_docs(hit.parsedContent, hit.meta.id)
    all_docs.extend(docs)
    # for para in text.split("\n\n"):
    #     if not text.strip():
    #         continue
    #     doc = {
    #         "meta": {
    #             "url": hit.meta.id,
    #             "@timestamp": datetime.utcnow()
    #         },
    #         "text": text
    #     }
        

document_store.write_documents(all_docs)

### Testing code ###

from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.reader.farm import FARMReader
from haystack.pipeline import ExtractiveQAPipeline
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
retriever = ElasticsearchRetriever(document_store=document_store)
pipeline = ExtractiveQAPipeline(reader, retriever)
query = 'What does AHRQ stand for?'
prediction = pipeline.run(query=query, top_k_retriever=10,top_k_reader=3)


document_store2 = ElasticsearchDocumentStore(
    host=get_hostname(os.environ.get("ELASTIC_URL")),
    username="elastic", 
    password=os.environ.get("ELASTIC_PASSWORD"),
    index="ahrq_qa"
)

retriever2 = ElasticsearchRetriever(document_store=document_store2)

pipeline2 = ExtractiveQAPipeline(reader, retriever2)

filters = [
    Q("term", siteDomain="www.ahrq.gov"), # just the www.ahrq.gov pages
    Q("term", mimeType__keyword="text/html"), # just the html page
    Q("exists", field="parsedContent"), # make sure there is text on page
    Q("range", tstamp={"lt": "2021-03-17"}),
]