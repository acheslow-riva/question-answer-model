import os

from app import create_app
from flask_script import Manager, Shell

a = create_app(os.environ.get("FLASK_CONFIG", 'default'))
manager = Manager(a)

def make_shell_context():
    return dict(app=a)

manager.add_command('shell', Shell(make_context=make_shell_context))

@manager.command
def dev():
    # import ptvsd
    # print('Attaching to ptvsd!!!!!!!!', flush=True)
    # ptvsd.enable_attach(address=("0.0.0.0", 5678))
    a.run(host='0.0.0.0', port=8000, use_reloader=False)

if __name__ == '__main__':
    manager.run()