TinyML Example: Anomaly Detection
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

You will need to install TensorFlow, Keras, and Jupyter Notebook on your desktop or laptop. [This guide](https://www.digikey.com/en/maker/projects/getting-started-with-machine-learning-using-tensorflow-and-keras/0746640deea84313998f5f95c8206e5b) will walk you through that process. 

Alternatively, you can use [Google Colab](https://colab.research.google.com/) to run a Jupyter Notebook instance in the cloud, however, loading files (e.g. training samples) will require you to upload them to Google Drive and write different code to import them into your program. [This guide](https://towardsdatascience.com/3-ways-to-load-csv-files-into-colab-7c14fcbdcb92) offers some tips on how to do that.

If you plan to collect data yourself, you will need an [Adafruit Feather Huzzah32](https://www.digikey.com/product-detail/en/adafruit-industries-llc/3591/1528-2514-ND/8119805), [Adafruit MSA301 accelerometer breakout board](https://www.digikey.com/products/en/development-boards-kits-programmers/evaluation-boards-expansion-boards-daughter-cards/797?k=msa301), battery, breadboard, and jumper wires.

Getting Started
---------------

Download this repository. Open **anomaly-detection-feature-analysis** in Jupyter Notebook and run it. Carefully look at the various plots to determine which features can be used to best discriminate between normal and anomalous operation.

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