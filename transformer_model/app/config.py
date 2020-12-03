import os

class Config:
    ELASTIC_URL = os.environ.get('ELASTIC_URL', 'localhost')
    ELASTIC_PORT = os.environ.get('ELASTIC_PORT', 9200)
    ELASTIC_USER = os.environ.get('ELASTIC_USER')
    ELASTIC_PASSWORD = os.environ.get('ELASTIC_PASSWORD')
    ELASTIC_INDEX = os.environ.get('ELASTIC_INDEX', 'ahrq')


class DevelopConfig(Config):
    DEBUG=True

class ProductionConfig(Config):
    DEBUG=False

config = {
    'develop': DevelopConfig,
    'production': ProductionConfig,
    'default': DevelopConfig
}