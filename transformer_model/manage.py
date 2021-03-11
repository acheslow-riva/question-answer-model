import os
import signal

from gevent import monkey, signal_handler
monkey.patch_all()

from app import create_app

app = create_app(os.getenv("FLASK_CONFIG", "default"))

def shutdown(server):
    server.log.writelines([f"Stopping {os.environ.get('HOSTNAME')} server\n"])
    server.stop()
    exit(0)

@app.cli.command("deploy")
def deploy():
    PORT = int(os.environ.get("PORT", 8000))
    from gevent.pywsgi import WSGIServer
    http_server = WSGIServer(('', PORT), app)
    signal_handler(signal.SIGTERM, shutdown, http_server)
    signal_handler(signal.SIGINT, shutdown, http_server)
    http_server.serve_forever()