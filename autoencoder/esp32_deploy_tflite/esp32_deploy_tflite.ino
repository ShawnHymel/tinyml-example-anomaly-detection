/**
 * Use TensorFlow Lite model on real accelerometer data to detect anomalies
 * 
 * Author: Shawn Hymel
 * Date: May 6, 2020
 * 
 * License: Beerware
 */

// Library includes
#include <Adafruit_MSA301.h>
#include <Adafruit_Sensor.h>

// Local includes
#include "fan_low_model.h"

// Import TensorFlow stuff
#include "TensorFlowLite.h"
#include "tensorflow/lite/experimental/micro/kernels/micro_ops.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/experimental/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/version.h"

// We need our utils functions for calculating MAD
extern "C" {
#include "utils.h"
};

// Set to 1 to output debug info to Serial, 0 otherwise
#define DEBUG 1

// Pins
constexpr int BUZZER_PIN = A1;

// Settings
constexpr int NUM_AXES = 3;           // Number of axes on accelerometer
constexpr int MAX_MEASUREMENTS = 128; // Number of samples to keep in each axis
constexpr float MAD_SCALE = 1.4826;   // Scale MAD to be inline with SciPy calcs
constexpr float THRESHOLD = 2e-05;    // Any MSE over this is an anomaly
constexpr int WAIT_TIME = 1000;       // ms between sample sets
constexpr int SAMPLE_RATE = 200;      // How fast to collect measurements (Hz)

// Globals
Adafruit_MSA301 msa;

// TFLite globals, used for compatibility with Arduino-style sketches
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;

  // Create an area of memory to use for input, output, and other TensorFlow
  // arrays. You'll need to adjust this by combiling, running, and looking
  // for errors.
  constexpr int kTensorArenaSize = 1 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
} // namespace
 
/*******************************************************************************
 * Main
 */
 
void setup() {

  // Initialize Serial port for debugging
#if DEBUG
  Serial.begin(115200);
  while (!Serial);
#endif

  // Initialize accelerometer
  if (!msa.begin()) {
#if DEBUG
    Serial.println("Failed to initialize MSA301");
#endif
    while (1);
  }

  // Configure buzzer pin
  pinMode(BUZZER_PIN, OUTPUT);

  // Set up logging (will report to Serial, even within TFLite functions)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure
  model = tflite::GetModel(fan_low_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version does not match Schema");
    while(1);
  }

  // Pull in only needed operations (should match NN layers)
  // Available ops:
  //  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/micro_ops.h
  static tflite::MicroMutableOpResolver micro_mutable_op_resolver;
  micro_mutable_op_resolver.AddBuiltin(
    tflite::BuiltinOperator_FULLY_CONNECTED,
    tflite::ops::micro::Register_FULLY_CONNECTED(),
    1, 3);

  // Build an interpreter to run the model
  static tflite::MicroInterpreter static_interpreter(
    model, micro_mutable_op_resolver, tensor_arena, kTensorArenaSize,
    error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while(1);
  }

  // Assign model input and output buffers (tensors) to pointers
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);


  
  
}

void loop() {

  float sample[MAX_MEASUREMENTS][NUM_AXES];
  float measurements[MAX_MEASUREMENTS];
  float mad[NUM_AXES];
  float y_val[NUM_AXES];
  float mse;
  TfLiteStatus invoke_status;
  
  // Timestamps for collecting samples
  static unsigned long timestamp = millis();
  static unsigned long prev_timestamp = timestamp;

  // Take a given time worth of measurements
  int i = 0;
  while (i < MAX_MEASUREMENTS) {
    if (millis() >= timestamp + (1000 / SAMPLE_RATE)) {
  
      // Update timestamps to maintain sample rate
      prev_timestamp = timestamp;
      timestamp = millis();

      // Take sample measurement
      msa.read();

      // Add readings to array
      sample[i][0] = msa.x_g;
      sample[i][1] = msa.y_g;
      sample[i][2] = msa.z_g;

      // Update sample counter
      i++;
    }
  }
  
  // For each axis, compute the MAD (scale up by 1.4826)
  for (int axis = 0; axis < NUM_AXES; axis++) {
    for (int i = 0; i < MAX_MEASUREMENTS; i++) {
      measurements[i] = sample[i][axis];
    }
    mad[axis] = MAD_SCALE * calc_mad(measurements, MAX_MEASUREMENTS);
  }

  // Print out MAD calculations
#if DEBUG
  Serial.print("MAD: ");
  for (int axis = 0; axis < NUM_AXES; axis++) {
    Serial.print(mad[axis], 7);
    Serial.print(" ");
  }
  Serial.println();
#endif

  // Copy MAD values to input buffer/tensor
  for (int axis = 0; axis < NUM_AXES; axis++) {
    model_input->data.f[axis] = mad[axis];
  }

  // Run inference
  invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed on input");
  }

  // Read predicted y value from output buffer (tensor)
  for (int axis = 0; axis < NUM_AXES; axis++) {
    y_val[axis] = model_output->data.f[axis];
  }

  // Calculate MSE between given and predicted MAD values
  mse = calc_mse(mad, y_val, NUM_AXES);

  // Print out result
#if DEBUG
  Serial.print("Inference result: ");
  for (int axis = 0; axis < NUM_AXES; axis++) {
    Serial.print(y_val[axis], 7);
  }
  Serial.println();
  Serial.print("MSE: ");
  Serial.println(mse, 7);
#endif

  // Compare to threshold
  if (mse > THRESHOLD) {
    digitalWrite(BUZZER_PIN, HIGH);
#if DEBUG
    Serial.println("DANGER!!!");
#endif
  } else {
    digitalWrite(BUZZER_PIN, LOW);
  }
#if DEBUG
  Serial.println();
#endif

  delay(WAIT_TIME);

}
