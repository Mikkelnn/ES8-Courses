#include <Arduino.h>
#include <Arduino_LSM6DS3.h>

/*
  Arduino LSM6DS3 - Simple Accelerometer
  Amended by Anders la Cour-Harbo for setting acc and gyro rates.

  This example reads the acceleration values from the LSM6DS3
  sensor and continuously prints them to the Serial Monitor
  or Serial Plotter.

  The circuit:
  - Arduino Uno WiFi Rev 2 or Arduino Nano 33 IoT

  created 10 Jul 2019
  by Riccardo Rizzo

  This example code is in the public domain.
*/

/* This "trick" is called inheritance. We make a new class that inherits everything from the old class.
   We can then add additional functionality to the new class. In this case, we add four functions to
   add the ability to change the sampling rate for the gyro and accelerometer.
*/
class IMUExtended : public LSM6DS3Class
{
public:
  IMUExtended(TwoWire &wire, uint8_t slaveAddress)
      : LSM6DS3Class{wire, slaveAddress}
  {
  }
  void SetAccGyroRate13Hz()
  {
    writeRegister(0x10, 0b00011000); // See the LSM6DS3 documentation for what to write to these registers.
    writeRegister(0x11, 0b00011100);
  }
  void SetAccGyroRate26Hz()
  {
    writeRegister(0x10, 0b00101000);
    writeRegister(0x11, 0b00101100);
  }
  void SetAccGyroRate52Hz()
  {
    writeRegister(0x10, 0b00111000);
    writeRegister(0x11, 0b00111100);
  }
  void SetAccGyroRate104Hz()
  {
    writeRegister(0x10, 0b01001000);
    writeRegister(0x11, 0b01001100);
  }
};

// Instantiate the new class instead of the old class.
IMUExtended myIMU{Wire, LSM6DS3_ADDRESS};

uint32_t Time{0};

void setup()
{
  Serial.begin(115200);
  while (!Serial)
    ;

  if (!myIMU.begin())
  {
    Serial.println("Failed to initialize IMU!");

    while (1)
      ;
  }

  myIMU.SetAccGyroRate13Hz();

  Serial.println("Acceleration in g's");
  Serial.println("X\tY\tZ");
}

void loop()
{
  float x{0.f};
  float y{0.f};
  float z{0.f};

  // myIMU.gyroscopeSampleRate();    // Note that this function does not return the rate, but just always 104 Hz
  // myIMU.accelerationSampleRate(); // Note that this function does not return the rate, but just always 104 Hz

  if (myIMU.accelerationAvailable())
  {
    Serial.print("Time: ");
    Serial.println(millis() - Time);
    Time = millis();

    myIMU.readAcceleration(x, y, z);

    Serial.print(x);
    Serial.print('\t');
    Serial.print(y);
    Serial.print('\t');
    Serial.println(z);
  }
}
