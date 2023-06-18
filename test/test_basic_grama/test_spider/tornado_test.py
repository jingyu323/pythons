import tornado.ioloop
import tornado.web

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello World")


def make_app():
    return tornado.web.Application([
        (r"/",MainHandler),
    ])



if __name__ == "__main__":
    print("sss")
    app =  make_app();
    app.listen(9999)
    tornado.ioloop.IOLoop.current().start()