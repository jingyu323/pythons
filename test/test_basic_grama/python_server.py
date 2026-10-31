from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        # 打印请求的URL参数
        query_components = parse_qs(self.path.split('?')[1])
        for key, values in query_components.items():
            print(f"{key}: {values[0]}")




if __name__ == '__main__':
    httpd = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
    print("Serving at http://localhost:8000")
    httpd.serve_forever()