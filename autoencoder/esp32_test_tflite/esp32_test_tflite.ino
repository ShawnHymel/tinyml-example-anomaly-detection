/**
 * Import sample and use TensorFlow Lite model to see if it's an anomaly.
 * 
 * Author: Shawn Hymel
 * Date: May 6, 2020
 * 
 * License: Beerware
 */

// Import TensorFlow stuff
#include "TensorFlowLite.h"
#include "tensorflow/lite/experimental/micro/kernels/micro_ops.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/experimental/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/version.h"

// Our model and samples
#include "fan_low_model.h"
#include "normal_sample.h"
#include "anomaly_sample.h"

// We need our utils functions for calculating MAD
extern "C" {
#include "md_utils.h"
};

// Settings
constexpr int MAX_MEASUREMENTS = 128; // Number of samples to keep in each axis
constexpr float MAD_SCALE = 1.4826;   // Scale MAD to be inline with SciPy calcs

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

  float measurements[MAX_MEASUREMENTS];
  float mad[normal_sample_dim2];
  unsigned long time_start;
  unsigned long time_end;
  float y_val[normal_sample_dim2];
  float mse;
  TfLiteStatus invoke_status;
  
  // Make some Serial
  Serial.begin(115200);
  while(!Serial);
  Serial.println("TensorFlow Lite test");
  Serial.println();

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

  // Get information about the memory area to use for the model's input
  // Supported data types:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/common.h#L226
  Serial.print("Number of dimensions: ");
  Serial.println(model_input->dims->size);
  Serial.print("Dim 1 size: ");
  Serial.println(model_input->dims->data[0]);
  Serial.print("Dim 2 size: ");
  Serial.println(model_input->dims->data[1]);
  Serial.print("Input type: ");
  Serial.println(model_input->type);

  /**
   * Perform inference on a normal sample
   */
  Serial.println();
  Serial.println("---Normal Sample---");

  // For timing
  time_start = micros();

  // For each axis, compute the MAD (scale up by 1.4826)
  for (int axis = 0; axis < normal_sample_dim2; axis++) {
    for (int i = 0; i < MAX_MEASUREMENTS; i++) {
      measurements[i] = normal_sample[i][axis];
    }
    mad[axis] = MAD_SCALE * calc_mad(measurements, MAX_MEASUREMENTS);
  }

  // For timing
  time_end = micros();

  // Print out MAD calculations
  Serial.print("MAD: ");
  for (int axis = 0; axis < normal_sample_dim2; axis++) {
    Serial.print(mad[axis], 7);
    Serial.print(" ");
  }
  Serial.println();
  Serial.print("Time to compute MAD: ");
  Serial.print(time_end - time_start);
  Serial.println(" us");

  // For timing
  time_start = micros();
  
  // Copy MAD values to input buffer/tensor
  for (int axis = 0; axis < normal_sample_dim2; axis++) {
    model_input->data.f[axis] = mad[axis];
  }

  // Run inference
  invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed on input");
  }

  // Read predicted y value from output buffer (tensor)
  for (int axis = 0; axis < normal_sample_dim2; axis++) {
    y_val[axis] = model_output->data.f[axis];
  }

  // Calculate MSE between given and predicted MAD values
  mse = calc_mse(mad, y_val, normal_sample_dim2);

  // For timing
  time_end = micros();

  // Print out result
  Serial.print("Inference result: ");
  for (int axis = 0; axis < normal_sample_dim2; axis++) {
    Serial.print(y_val[axis], 7);
  }
  Serial.println();
  Serial.print("MSE: ");
  Serial.println(mse, 7);
  Serial.print("Time to perform inference and MSE: ");
  Serial.print(time_end - time_start);
  Serial.println(" us");

  /**
   * Perform inference on an anomaly sample
   */
  Serial.println();
  Serial.println("---Anomaly Sample---");

  // For timing
  time_start = micros();

  // For each axis, compute the MAD (scale up by 1.4826)
  for (int axis = 0; axis < anomaly_sample_dim2; axis++) {
    for (int i = 0; i < MAX_MEASUREMENTS; i++) {
      measurements[i] = anomaly_sample[i][axis];
    }
    mad[axis] = MAD_SCALE * calc_mad(measurements, MAX_MEASUREMENTS);
  }

  // For timing
  time_end = micros();

  // Print out MAD calculations
  Serial.print("MAD: ");
  for (int axis = 0; axis < anomaly_sample_dim2; axis++) {
    Serial.print(mad[axis], 7);
    Serial.print(" ");
  }
  Serial.println();
  Serial.print("Time to compute MAD: ");
  Serial.print(time_end - time_start);
  Serial.println(" us");

  // For timing
  time_start = micros();
  
  // Copy MAD values to input buffer/tensor
  for (int axis = 0; axis < anomaly_sample_dim2; axis++) {
    model_input->data.f[axis] = mad[axis];
  }

  // Run inference
  invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed on input");
  }

  // Read predicted y value from output buffer (tensor)
  for (int axis = 0; axis < normal_sample_dim2; axis++) {
    y_val[axis] = model_output->data.f[axis];
  }

  // Calculate MSE between given and predicted MAD values
  mse = calc_mse(mad, y_val, normal_sample_dim2);

  // For timing
  time_end = micros();

  // Print out result
  Serial.print("Inference result: ");
  for (int axis = 0; axis < normal_sample_dim2; axis++) {
    Serial.print(y_val[axis], 7);
  }
  Serial.println();
  Serial.print("MSE: ");
  Serial.println(mse, 7);
  Serial.print("Time to perform inference and MSE: ");
  Serial.print(time_end - time_start);
  Serial.println(" us");
}

void loop() {
  // put your main code here, to run repeatedly:

}
