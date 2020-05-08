import sys
from os import listdir
from os.path import join
import numpy as np
import scipy as sp
import random
from scipy import stats

# Print versions
print('Python ' + sys.version)
print('Numpy ' + np.__version__)
print('SciPy ' + sp.__version__)

# Settings
dataset_path = '../datasets/ceiling-fan-deploy'  # Directory where raw accelerometer data is stored
normal_op_list = ['fan_0_low-deploy']
val_ratio = 0.2             # Percentage of samples that should be held for validation set
sensor_sample_rate = 200    # Hz
sample_time = 0.64           # Time (sec) length of each sample
max_measurements = int(sample_time * sensor_sample_rate)
md_file_name = 'models/md_model_test-deploy'   # Mahalanobis Distance model arrays (.npz will be added)
print('Max measurements per file:', max_measurements)

# Create list of filenames
def createFilenameList(op_list):
    
    # Extract paths and filenames in each directory
    op_filenames = []
    num_samples = 0
    for index, target in enumerate(op_list):
        samples_in_dir = listdir(join(dataset_path, target))
        samples_in_dir = [join(dataset_path, target, sample) for sample in samples_in_dir]
        op_filenames.append(samples_in_dir)
    
    # Flatten list
    return [item for sublist in op_filenames for item in sublist]

# Create normal and anomaly filename lists
normal_op_filenames = createFilenameList(normal_op_list)
print('Number of normal samples:', len(normal_op_filenames))

# Shuffle lists
random.shuffle(normal_op_filenames)

# Calculate validation set size
val_set_size = int(len(normal_op_filenames) * val_ratio)

# Break dataset apart into train, validation, and test sets
num_samples = len(normal_op_filenames)
filenames_val = normal_op_filenames[:val_set_size]
filenames_train = normal_op_filenames[val_set_size:]

# Print out number of samples in each set
print('Number of training samples:', len(filenames_train))
print('Number of validation samples:', len(filenames_val))

# Check that our splits add up correctly
assert(len(filenames_train) + len(filenames_val)) == num_samples

# Function: extract specified features (variances, MAD) from sample
def extract_features(sample, max_measurements=0, scale=1):
    
    features = []
    
    # Truncate sample
    if max_measurements == 0:
        max_measurements = sample.shape[0]
    sample = sample[0:max_measurements]
    
    # Scale sample
    sample = scale * sample
    
    # Median absolute deviation (MAD)
    features.append(stats.median_absolute_deviation(sample))
    
    return np.array(features).flatten()

# Function: loop through filenames, creating feature sets
def create_feature_set(filenames):
    x_out = []
    for file in filenames:
        sample = np.genfromtxt(file, delimiter=',')
        features = extract_features(sample, max_measurements)
        x_out.append(features)
        
    return np.array(x_out)

# Create training, validation, and test sets
x_train = create_feature_set(filenames_train)
print('Extracted features from training set. Shape:', x_train.shape)
x_val = create_feature_set(filenames_val)
print('Extracted features from validation set. Shape:', x_val.shape)

# Use training data (normal) to calculate mean and covariance matrices
model_mu = np.mean(x_train, axis=0)
model_cov = np.cov(x_train.T)
print('Mean:', model_mu)
print('Covariance:')
print(model_cov)

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

# Calculate the mahalanobis distance for each validation sample
normal_mds = mahalanobis(x_val, model_mu, model_cov)
print('Average MD for normal validation set:', np.average(normal_mds))
print('Standard deviation of MDs for normal validation set:', np.std(normal_mds))
print('Recommended threshold (3x std dev + avg):', (3*np.std(normal_mds)) + np.average(normal_mds))

# Save Mahalanobis Distance model
np.savez(md_file_name + '.npz', model_mu=model_mu, model_cov=model_cov)
print('Mahalanobis Distance model saved to:', md_file_name + '.npz')