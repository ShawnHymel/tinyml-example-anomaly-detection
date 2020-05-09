/**
 * Import sample and use Mahalanobis Distance to see if it's an anomaly.
 * 
 * Author: Shawn Hymel
 * Date: May 5, 2020
 * 
 * License: Beerware
 */

#include "md_model-moving.h"
#include "normal_sample.h"
#include "anomaly_sample.h"

extern "C" {
#include "utils.h"
};

// Settings
constexpr int MAX_MEASUREMENTS = 128; // Number of samples to keep in each axis
constexpr float MAD_SCALE = 1.4826;   // Scale MAD to be inline with SciPy calcs

/*******************************************************************************
 * Main
 */
void setup() {

  float mahal;
  float measurements[MAX_MEASUREMENTS];
  float mad[normal_sample_dim2];
  unsigned long time_start;
  unsigned long time_end;

  // Start some serial
  Serial.begin(115200);
  while(!Serial);
  Serial.println("Mahalanobis Distance test");
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

  // Calculate Mahalanobis distance of normal sample
  time_start = micros();
  mahal = mahalanobis(mad, model_mu, *model_inv_cov, model_mu_dim1);
  time_end = micros();
  Serial.print("Mahalanobis distance: ");
  Serial.println(mahal, 7);
  Serial.print("Time to compute MD: ");
  Serial.print(time_end - time_start);
  Serial.println(" us");

  // Repeat the process for the anomaly sample
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
  for (int axis = 0; axis < normal_sample_dim2; axis++) {
    Serial.print(mad[axis], 7);
    Serial.print(" ");
  }
  Serial.println();
  Serial.print("Time to compute MAD: ");
  Serial.print(time_end - time_start);
  Serial.println(" us");

  // Calculate Mahalanobis distance of normal sample
  time_start = micros();
  mahal = mahalanobis(mad, model_mu, *model_inv_cov, model_mu_dim1);
  time_end = micros();
  Serial.print("Mahalanobis distance: ");
  Serial.println(mahal, 7);
  Serial.print("Time to compute MD: ");
  Serial.print(time_end - time_start);
  Serial.println(" us");
}

void loop() {
  // put your main code here, to run repeatedly:

}
