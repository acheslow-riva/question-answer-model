import os

class Config:
    ELASTIC_HOST = os.environ.get('ELASTIC_HOST', 'localhost')
    ELASTIC_PORT = os.environ.get('ELASTIC_PORT', 9200)
    ELASTIC_URL = os.environ.get('ELASTIC_URL', f"http://{ELASTIC_HOST}:{ELASTIC_PORT}")
    ELASTIC_USER = os.environ.get('ELASTIC_USER', 'elastic')
    ELASTIC_PASSWORD = os.environ.get('ELASTIC_PASSWORD')
    ELASTIC_INDEX = os.environ.get('ELASTIC_INDEX', 'ahrq')


class LocalConfig(Config):
    DEBUG=True
    USE_TRACED_MODEL=False  

class DevelopConfig(Config):
    DEBUG=True
    USE_TRACED_MODEL=True

class ProductionConfig(Config):
    DEBUG=False
    USE_TRACED_MODEL=True

config = {
    'develop': DevelopConfig,
    'production': ProductionConfig,
    'default': LocalConfig
}