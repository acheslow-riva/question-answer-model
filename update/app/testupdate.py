from datetime import datetime, timedelta
import logging
import os
import sys
from urllib import parse

from elasticsearch_dsl.connections import connections
from elasticsearch_dsl import A, Q, Search
es = connections.create_connection(
    hosts=[os.environ.get("ELASTIC_URL")],
    http_auth=("elastic", os.environ.get("ELASTIC_PASSWORD"))
)
if es.ping():
    print("Hello")
else:
    print("Oh no")