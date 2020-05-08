import os
import argparse
import json
import threading
import time
import numpy as np
from os.path import join
from io import BytesIO
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler
import scipy as sp
from scipy import stats

# Settings
DEFUALT_PORT = 1337
MODELS_PATH = 'models'
MD_MODEL_FILE = 'md_model_test-deploy'  # .npz will be added
MAX_MEASUREMENTS = 128      # Truncate measurements to this number
ANOMALY_THRESHOLD = 9.0    # An MD over this will be considered an anomaly

# Global flag
server_ready = 0

################################################################################
# Functions

# Function: extract specified features (MAD) from sample
def extract_features(sample, max_measurements=0):
    
    features = []
    
    # Truncate sample
    if max_measurements == 0:
        max_measurements = sample.shape[0]
    sample = sample[0:max_measurements]
    
    # Median absolute deviation (MAD)
    features.append(stats.median_absolute_deviation(sample))

    return np.array(features).flatten()

# Calculate mahalanobis distance of x from group described by mu, cov
# Based on: https://www.machinelearningplus.com/statistics/mahalanobis-distance/
def mahalanobis(x, mu, cov):
    x_minus_mu = x - mu
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    if mahal.shape == ():
        return mahal
    else:
        return mahal.diagonal()

# Decode string to JSON and save measurements in a file
def parseSamples(json_str):

    # Create a browsable JSON document
    try:
        json_doc = json.loads(json_str)
    except Exception as e:
        print('ERROR: Could not parse JSON |', str(e))
        return
    
    # Parse sample
    sample = []
    num_meas = len(json_doc['x'])
    for i in range(0, num_meas):
        sample.append([float(json_doc['x'][i]),
                        float(json_doc['y'][i]),
                        float(json_doc['z'][i])])

    # Calculate MAD for each axis
    feature_set = extract_features(np.array(sample), 
                                    max_measurements=MAX_MEASUREMENTS)

    # Compute Mahalnobis Distance between sample and model mean
    mahal = mahalanobis(feature_set, model_mu, model_cov)
    print("Mahalanobis Distance:", mahal)

    # Compare to threshold to see if we have an anomaly
    if mahal > ANOMALY_THRESHOLD:
        print("ANOMALY DETECTED!")
    else:
        print("Normal")

    return

# Handler class for HTTP requests
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_GET(self):

        # Tell client if server is ready for a new sample
        self.send_response(200)
        self.end_headers()
        self.wfile.write(str(server_ready).encode())

    def do_POST(self):

        # Read message
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)

        # Respond with 204 "no content" status code
        self.send_response(204)
        self.end_headers()

        # Decode JSON and compute MSE
        file_num = parseSamples(body.decode('ascii'))

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
parser.add_argument('-p', action='store', dest='port', type=int,
                    default=DEFUALT_PORT, help='Port number for server')
args = parser.parse_args()
port = args.port

# Print versions
print('Numpy ' + np.__version__)
print('SciPy ' + sp.__version__)

# Load model
with np.load(join(MODELS_PATH, MD_MODEL_FILE) + '.npz') as data:
    model_mu = data['model_mu']
    model_cov = data['model_cov']
print(model_mu)
print(model_cov)

# Create server
handler = partial(SimpleHTTPRequestHandler)
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
while True:
    pass
print('Server shutting down')
server.shutdown()
server_thread.stop()
server_thread.join()
