from datetime import datetime, timedelta
import logging
import os
import sys
from urllib import parse

from elasticsearch_dsl.connections import connections
from elasticsearch_dsl import A, Q, Search
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore

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


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


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
                    "timestamp": datetime.utcnow(),
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

if __name__ == "__main__":
    es = connections.create_connection(
        hosts=[os.environ.get("ELASTIC_URL")],
        http_auth=("elastic", os.environ.get("ELASTIC_PASSWORD"))
    )
    assert es.ping()
    logging.info("Connected to elasticsearch")
    new_install = not es.indices.exists(INDEX)
    if new_install: logging.info("New installation. Ingesting all ahrq data.")
    document_store = ElasticsearchDocumentStore(
        host=get_hostname(os.environ.get("ELASTIC_URL")),
        username="elastic", 
        password=os.environ.get("ELASTIC_PASSWORD"),
        index=INDEX
    )

    ## Get start date:
    start_date = datetime(2021, 3, 17)
    a = A("max", field="timestamp")
    r = Search(index=INDEX)
    s = Search(index="webpages")
    r.aggs.bucket("latest_date", a)
    start_date = r[0:0].execute().aggs.latest_date
    if start_date.value:
        start_date = start_date.value_as_string
    else: # new index
        start_date = "2020-12-05"

    # Find all documents with detectedDate >= start_date
    # OR documents that don't have detectedDate but whose
    # tstamp >= start_date
    if new_install:
        filters = [
            Q("term", siteDomain="www.ahrq.gov"), # just the www.ahrq.gov pages
            Q("term", mimeType__keyword="text/html"), # just the html page
            Q("exists", field="parsedContent"), # make sure there is text on page
        ]
        q = Q("bool", filter=filters)
    else:
        filters = [
            Q("bool",
                filter=[
                    Q("term", siteDomain="www.ahrq.gov"), # just the www.ahrq.gov pages
                    Q("term", mimeType__keyword="text/html"), # just the html page
                    Q("exists", field="parsedContent"), # make sure there is text on page
                    Q("bool", should=[
                        Q("bool", filter=[
                            Q("range", tstamp={"gte": start_date}),
                            Q("bool", must_not=Q("exists", field="detectedDate")),
                        ]),
                        Q("bool", filter=[
                            Q("range", detectedDate={"gte": start_date}),
                        ])
                    ])
                ]
            )
        ]
        q = Q("bool", filter=filters)
        # Delete ids if they already exist in the index. First, get ids:
        r = Search(index=INDEX)
        ids = [hit.meta.id for hit in s.extra(_source=False).query(q).scan()]
        # Now we run an delete by query to delete urls matching these ids:
        for chunk_id in chunks(ids, 20):
            p = Q("terms", url=chunk_id)
            logging.info(f"Deleting docs: {chunk_id}")
            r.query(p).params(refresh=True).delete()
    
    doc_count= s.query(q).count()
    if not doc_count:
        logging.info("No new documents to add.")
        exit(0)

    q = Q("bool", filter=filters)
    all_docs = []
    for hit in s.extra(_source="parsedContent").query(q).scan():
        docs = text_to_docs(hit.parsedContent, hit.meta.id)
        all_docs.extend(docs)
    document_store.write_documents(all_docs)
    logging.info(f"Added {len(all_docs)} new paragraphs from {doc_count} documents.")
