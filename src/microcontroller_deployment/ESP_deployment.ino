#include <Arduino.h>
#include "MLP_model.h"
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

#define ARENA_SIZE 100000

Eloquent::TF::Sequential<2, ARENA_SIZE> tf;

void setupTensorFlow() {
    tf.setNumInputs(5);
    tf.setNumOutputs(128);

    tf.resolver.AddFullyConnected();
    tf.resolver.AddRelu();

    if (!tf.begin(model).isOk()) {
        Serial.println("Error: TensorFlow Lite initialization failed!");
        while (true);
    }
}

void setup() {
    Serial.begin(115200);
    while(!Serial) delay(100);
    setupTensorFlow();
}

void loop() {
  if (Serial.available() >= 20) {
    float receivedData[5];
    // Read 20 bytes (5 floats) directly into the array
    // Expected order from Python script:
    // [0]: Irms
    // [1]: Real Power (P)
    // [2]: Power Factor (PF)
    // [3]: Apparent Power (S)
    // [4]: Reactive Power (Q)
    Serial.readBytes((char*)receivedData, 20);

    float irms          = receivedData[0];
    float realPower     = receivedData[1];
    float powerFactor   = receivedData[2];
    float apparentPower = receivedData[3];
    float reactivePower = receivedData[4];

    // Model input features: [Irms, RealPower, PowerFactor, ApparentPower, ReactivePower]
    float x[5] = {irms, realPower, powerFactor, apparentPower, reactivePower};

    tf.predict(x);
    // Print the predicted class index
    Serial.print(tf.classification);
    Serial.print(";");
    Serial.println(0); // Placeholder or secondary output
  }
}
