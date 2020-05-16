# TinyML Example: Anomaly Detection

This project is an example demonstrating how to use Python to train two different machine learning models to detect anomalies in an electric motor. The first model relies on the classic machine learning technique of Mahalanobis distance. The second model is an autoencoder neural network created with TensorFlow and Keras.

Data was captured using an ESP32 and MSA301 3-axis accelerometer taped to a ceiling fan. Each sample is about 200 samples of all 3 axes captured over the course of 1 second. Fan was run at multiple speeds (off, low, medium, high) with and without a weight. 1 "weight" is one US quarter taped to one of the fan's blades to create an offset motion. All raw data is stored in the ceiling-fan-dataset directory.

The full articles that explain how these programs work and how to use them can be found here:
* [Edge AI Anomaly Detection Part 1 - Data Collection](https://www.digikey.com/en/maker/projects/edge-ai-anomaly-detection-part-1-data-collection/7bb112f76ef644edaedc5e08dba5faae)
* [Edge AI Anomaly Detection Part 2 - Feature Extraction and Model Training](https://www.digikey.com/en/maker/projects/edge-ai-anomaly-detection-part-2-feature-extraction-and-model-training/70927a6e439b49bea7305953a3c9bfff)
* [Edge AI Anomaly Detection Part 3 - Machine Learning on Raspberry Pi](https://www.digikey.com/en/maker/projects/edge-ai-anomaly-detection-part-3-machine-learning-on-raspberry-pi/af9dd958b23d4ea1b40bc3cc060ef8c9)
* [Edge AI Anomaly Detection Part 4 - Machine Learning Models on Arduino](https://www.digikey.com/en/maker/projects/edge-ai-anomaly-detection-part-4-machine-learning-models-on-arduino/afacfc3dbaf24c6c94a55c4afae1afb2)

Here are the accompanying YouTube videos that explain how to use these programs and some of the theory behind them:
* [Edge AI Anomaly Detection Part 1: Data Collection](https://www.youtube.com/watch?v=Bxd7W1I-tq4)
* *Rest coming soon*

![ESP32 on ceiling fan for anomaly detection](https://raw.githubusercontent.com/ShawnHymel/tflite-anomaly-detection-example/master/images/fan-anomaly-detection-cover.jpg)

### Prerequisites

You will need to install TensorFlow, Keras, and Jupyter Notebook on your desktop or laptop. [This guide](https://www.digikey.com/en/maker/projects/getting-started-with-machine-learning-using-tensorflow-and-keras/0746640deea84313998f5f95c8206e5b) will walk you through that process. 

Alternatively, you can use [Google Colab](https://colab.research.google.com/) to run a Jupyter Notebook instance in the cloud, however, loading files (e.g. training samples) will require you to upload them to Google Drive and write different code to import them into your program. [This guide](https://towardsdatascience.com/3-ways-to-load-csv-files-into-colab-7c14fcbdcb92) offers some tips on how to do that.

If you plan to collect data yourself, you will need an [Adafruit Feather Huzzah32](https://www.digikey.com/product-detail/en/adafruit-industries-llc/3591/1528-2514-ND/8119805), [Adafruit MSA301 accelerometer breakout board](https://www.digikey.com/products/en/development-boards-kits-programmers/evaluation-boards-expansion-boards-daughter-cards/797?k=msa301), battery, breadboard, and jumper wires.

For deployment, you will want to install TensorFlow Lite on your Raspberry Pi by following [these instructions](https://www.tensorflow.org/lite/guide/python). For Arduino, you will want to install the [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers) Arduino library.

### Getting Started

Please read the tutorials above for full documentation on how to use the scripts found in this project.

In general, you will want to perform the following steps:

1. Collect data samples
2. Analyze data for good features
3. Train one or more machine learning models
4. Deploy machine learning model to end system

#### Collect Data

Download this repository. Connect an MSA301 accelerometer breakout board to the ESP32 Feather. Open the [data_collection/esp32_accel_post/esp32_accel_post.ino](https://github.com/ShawnHymel/tinyml-example-anomaly-detection/blob/master/data_collection/esp32_accel_post/esp32_accel_post.ino) Arduino sketch and change the WiFi SSID, password, and server IP address. Upload to your ESP32. Place your ESP32 and accelerometer on the system you wish to monitor (e.g. ceiling fan).

Run [data_collection/http_accel_server.py](https://github.com/ShawnHymel/tinyml-example-anomaly-detection/blob/master/data_collection/http_accel_server.py) on your server computer to collect data. Repeat this process for however many normal and anomaly states you wish to collect data from. Recommend at least 200 sample files per state.

#### Analyze Data

Open [data_collection/anomaly-detection-feature-analysis](https://github.com/ShawnHymel/tinyml-example-anomaly-detection/blob/master/data_collection/anomaly-detection-feature-analysis.ipynb) in Jupyter Notebook and run it. Change the dataset_path to point to wherever you collected your sample files. Change the op_lists to be the names of the directories in that dataset path. Note that by default, the dataset path is set to datasets/ in this repository, which you are welcome to use (although it might not be indicative of your particular system).

Carefully look at the various plots to determine which features can be used to best discriminate between normal and anomalous operation.

#### Train Machine Learning Models

Run **anomaly-detection-training-mahalanobis-distance** to see how to create a mathematical model that can be used to find anomalies in the median absolute deviation (MAD) by calculating the Mahalanobis distance between new samples and the group of "normal" samples.

Run **anomaly-detection-training-autoencoder** to see how to create a neural network that finds anomalies in the median absolute deviation (MAD) in new samples. The model is trained on a set of "normal" samples collected from the ceiling fan.

Run **anomaly-detection-tflite-conversion** to create a TensorFlow Lite model from the .h5 Keras model (created in the anomaly-detection-training-autoencoder script).

Collecting Data
---------------

If you wish to collect your own data, you will need to connect an MSA301 accelerometer to an ESP32 (I used the Adafruit Feather Huzzah32) via I2C. Open the **esp32_accel_post** sketch and change the WiFi credentials and server IP address to match your computer's IP address. Upload the sketch to the ESP32.

Attach a battery to the ESP32 and secure it (using something like tape) to your electric motor (I used a ceiling fan).

Start **http-accel-server.py** with the following arguments:

```
python http-accel-server.py -d <output directory where samples are stored> -p <port, such as 1337> -t <time to run server; something like 2400 seconds seems to work well>
```

You'll want to run the collection process for each operating mode of your motor. For a ceiling fan, that's off, low, medium, and high. It can also help to collect some "anomaly" data in a separate folder. For example, tape a coin to a ceiling fan blade and run the collection process again with all operating modes (except for "off").

License
-------

All code in this repository, unless otherwise specified, is for demonstration purposes and licensed under [Beerware](https://en.wikipedia.org/wiki/Beerware).

Distributed as-is; no warranty is given.