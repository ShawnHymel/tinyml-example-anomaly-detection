TensorFlow Lite Anomaly Detection Example
========

This project is an example demonstrating how to use Python to train two different machine learning models to detect anomalies in an electric motor. The first model relies on the classic machine learning technique of Mahalanobis distance. The second model is an autoencoder neural network created with TensorFlow and Keras.

Data was captured using an ESP32 and MSA301 3-axis accelerometer taped to a ceiling fan. Each sample is about 200 samples of all 3 axes captured over the course of 1 second. Fan was run at multiple speeds (off, low, medium, high) with and without a weight. 1 "weight" is one US quarter taped to one of the fan's blades to create an offset motion. All raw data is stored in the ceiling-fan-dataset directory.

The full articles that explain how these programs work and how to use them can be found here:
**Coming Soon**

Here are the accompanying videos that explain how to use these programs and some of the theory behind them:
**Coming Soon**

![ESP32 on ceiling fan for anomaly detection](https://raw.githubusercontent.com/ShawnHymel/tflite-anomaly-detection-example/master/images/fan-anomaly-detection-cover.jpg)

Prerequisites
--------------

You will need to install TensorFlow, Keras, and Jupyter Notebook on your desktop or laptop. (This guide)[https://www.digikey.com/en/maker/projects/getting-started-with-machine-learning-using-tensorflow-and-keras/0746640deea84313998f5f95c8206e5b] will walk you through that process. 

Alternatively, you can use [Google Colab](https://colab.research.google.com/) to run a Jupyter Notebook instance in the cloud, however, loading files (e.g. training samples) will require you to upload them to Google Drive and write different code to import them into your program. [This guide](https://towardsdatascience.com/3-ways-to-load-csv-files-into-colab-7c14fcbdcb92) offers some tips on how to do that.

If you plan to collect data yourself, you will need an [Adafruit Feather Huzzah32](https://www.digikey.com/product-detail/en/adafruit-industries-llc/3591/1528-2514-ND/8119805), [Adafruit MSA301 accelerometer breakout board](https://www.digikey.com/products/en/development-boards-kits-programmers/evaluation-boards-expansion-boards-daughter-cards/797?k=msa301), battery, breadboard, and jumper wires.

Getting Started
---------------

Download this repository. Open **anomaly-detection-feature-analysis** in Jupyter Notebook and run it. Carefully look at the various plots to determine which features can be used to best discriminate between normal and anomalous operation.



License
-------

All code in this repository, unless otherwise specified, is for demonstration purposes and licensed under [Beerware](https://en.wikipedia.org/wiki/Beerware).

Distributed as-is; no warranty is given.