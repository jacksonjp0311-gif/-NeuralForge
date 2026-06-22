"""Simple HTTP server for the NeuralForge dashboard. Serves evo_results.json and the dashboard HTML."""
import http.server, json, os, threading, time
from urllib.parse import urlparse

PORT = 9876
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/evo-results':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            data_path = os.path.join(DATA_DIR, 'evo_results.json')
            if os.path.exists(data_path):
                with open(data_path, 'r') as f:
                    self.wfile.write(f.read().encode())
            else:
                self.wfile.write(json.dumps({"status": "no_data"}).encode())
        elif parsed.path == '/' or parsed.path == '/dashboard':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            dash_path = os.path.join(DATA_DIR, 'dashboard_live.html')
            if os.path.exists(dash_path):
                with open(dash_path, 'r') as f:
                    self.wfile.write(f.read().encode())
            else:
                self.wfile.write(b'Dashboard not found')
        else:
            super().do_GET()

    def log_message(self, format, *args):
        pass  # Suppress logs

def start_server():
    server = http.server.HTTPServer(('127.0.0.1', PORT), Handler)
    print(f"NeuralForge Dashboard running at http://127.0.0.1:{PORT}/dashboard")
    print(f"API endpoint: http://127.0.0.1:{PORT}/api/evo-results")
    server.serve_forever()

if __name__ == '__main__':
    start_server()
