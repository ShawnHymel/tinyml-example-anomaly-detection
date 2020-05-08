# Based on: https://blog.anvileight.com/posts/simple-python-http-server/

import os
import argparse
import json
import threading
import time
from io import BytesIO
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler

# Settings
DEFUALT_PORT = 1337
NUM_FILE_CHARS = 4      # 10^NUM files (e.g. 2 chars = 100 files)

# Global flag
server_ready = 0
file_num = 0

################################################################################
# Functions

# Decode string to JSON and save measurements in a file
def parseSamples(json_str, dir, file_num):

    # Create a browsable JSON document
    json_doc = json.loads(json_str)

    # Find next filename
    print('finding file')
    cnt = file_num
    print('starting with', cnt)
    while cnt < 10**NUM_FILE_CHARS:
        file_name = str(cnt).zfill(NUM_FILE_CHARS) + '.csv'
        file_path = os.path.join(dir, file_name)
        if not os.path.exists(file_path):
            break
        cnt += 1
    if cnt >= 10**NUM_FILE_CHARS:
        print('ERROR: Directory full')
        return file_num

    # Write to file
    print('Creating file:', file_path)
    with open(file_path, mode='w', encoding='utf-8') as f:
        num_meas = len(json_doc['x'])
        print('Writing samples', num_meas)
        try:
            for i in range(0, num_meas):
                f.write(str(json_doc['x'][i]) + ', ')
                f.write(str(json_doc['y'][i]) + ', ')
                f.write(str(json_doc['z'][i]))
                f.write('\n')
        except Exception as e:
            print('ERROR: Cannot write to file.', str(e))
            return file_num
    
    return cnt

# Handler class for HTTP requests
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def __init__(self, dir, *args, **kwargs):
        self.dir = dir
        self.file_num = 0
        super().__init__(*args, **kwargs)

    def do_GET(self):

        # Tell client if server is ready for a new sample
        self.send_response(200)
        self.end_headers()
        self.wfile.write(str(server_ready).encode())

    def do_POST(self):

        # Not a fan of this, but I couldn't find a better way to store a
        # value between calls of a callback
        global file_num

        # Read message
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)

        # Respond with 204 "no content" status code
        self.send_response(204)
        self.end_headers()

        # Decode JSON and save to file
        print('giving:', file_num)
        file_num = parseSamples(body.decode('ascii'), 
                                        self.dir, 
                                        file_num)
        print('ret:', file_num)

# Server thread
class ServerThread(threading.Thread):
    
    def __init__(self, *args, **kwargs):
        super(ServerThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def is_stopped(self):
        return self._stop_event.is_set()

################################################################################
# Main

# Parse arguments
parser = argparse.ArgumentParser(description='Server that saves data from' +
                                    'IoT sensor node.')
parser.add_argument('-d', action='store', dest='out_dir', type=str, 
                    required=True, help='Directory where samples are stored')
parser.add_argument('-p', action='store', dest='port', type=int,
                    default=DEFUALT_PORT, help='Port number for server')
parser.add_argument('-t', action='store', dest='record_time', type=float,
                    default=0, help='Time (in seconds) to record samples ' +
                    '(0 = run forever)')
args = parser.parse_args()
out_dir = args.out_dir
port = args.port
record_time = args.record_time

# If directory does not exist, create it
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# Create server
handler = partial(SimpleHTTPRequestHandler, out_dir)
server = HTTPServer(('', port), handler)
server_addr = server.socket.getsockname()
print('Server running at: ' + str(server_addr[0]) + ':' + 
        str(server_addr[1]))

# Create thread running server
server_thread = ServerThread(name='server_daemon',
                            target=server.serve_forever)
server_thread.daemon = True
server_thread.start()

# Store samples for given time
server_ready = 1
rec_timestamp = time.time()
if record_time == 0:
    while True:
        pass
else:
    while time.time() < rec_timestamp + record_time:
        pass
print('Server shutting down')
server.shutdown()
server_thread.stop()
server_thread.join()
