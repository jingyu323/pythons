from flask import Flask

from test2 import test_share_var2
app = Flask(__name__)


@app.route('/')
def hello_world():
    test_share_var2.share_var2()

    print("main share valur is :"+test_share_var2.share_var2)

    return 'Hello World!'


if __name__ == '__main__':
    app.run()
