/*  SparkFun ACS712 and ACS723 Demo
    Created by George Beckstein for SparkFun
    4/30/2017
    Updated by SFE
    6/14/2018

    Uses an Arduino to set up the ACS712 and ACS723 Current Sensors
    See the tutorial at: https://learn.sparkfun.com/tutorials/current-sensor-breakout-acs723-hookup-guide

    Parts you may need:
    - 100 Ohm, 1/2W or greater resistor OR two 220 Ohm 1/4 resistors in parallel
    - ACS712 Breakout with on-board amplifier or ACS723 Current Sensor (Low Current)

    Optional equipment:
    - Oscilloscope
    - Multimeter (or two)
    - A power supply with current limiting/constant current would be handy to calibrate the device without using resistors
*/

const int analogInPin = A0;

// Number of samples to average the reading over
// Change this to make the reading smoother... but beware of buffer overflows!
const long avgSamples = 50;
const int card_on_thresh = 1;
long sensorValue = 0;
long cardOn = 0;
bool vacuumOff = true;
float sensitivity = 100.0 / 500.0; //100mA per 500mV = 0.2
float Vref = 2500; // Output voltage with no current: ~ 2500mV or 2.5V

void setup() {
  // initialize serial communications at 9600 bps:
  Serial.begin(115200);
  pinMode(12, OUTPUT);
  pinMode(11, OUTPUT);
}

void loop() {
  unsigned long min = 10000;
  unsigned long max = 0;
  double avg = 0;
  for (int j = 0; j < avgSamples; j++) {
    long current = analogRead(analogInPin) - 500;
    if(current < min){
      min = current;
    }  
    if(current > max){
      max = current;
    }
    avg += current;
    delay(2);
  }
  avg /= avgSamples;    
  // wait 2 milliseconds before the next loop
  // for the analog-to-digital converter to settle
  // after the last reading:
  // This will calculate the actual current (in mA)
  // Using the Vref and sensitivity settings you configure
  float current = avg;

  // This is the raw sensor value, not very useful without some calculations
  //Serial.print(sensorValue);
// Vacuum off:          6 avg(8,10) 13
// Vacuum on:           6 avg(15,17) 37
// Vacuum sucking card: 6 avg(16,20) 41
  /*************************************************************************************
   * Step 1.)
   * Uncomment and run the following code to set up the baseline voltage 
   * (the voltage with 0 current flowing through the device).
   * Make sure no current is flowing through the IP+ and IP- terminals during this part!
   * 
   * The output units are in millivolts. Use the Arduino IDE's Tools->Serial Plotter
   * To see a plot of the output. Adjust the Vref potentiometer to set the reference
   * voltage. This allows the sensor to output positive and negative currents!
   *************************************************************************************/
  int vthresh = 3;
  if (avg >= 8 && 10+vthresh >= avg && max <= 13+vthresh) {
    vacuumOff = true;
    cardOn = 0;
    // Serial.println("VACUUM OFF ");  
  } else if (avg >= 15 && 17+vthresh >= avg && max <= 37+vthresh) {
    vacuumOff = false;
    // Serial.println("VACUUM ON ");
  } else if (avg >= 16 && 20+vthresh >= avg && max <= 41+vthresh) {
    vacuumOff = false;
    cardOn += 1;
  }

  if(cardOn > card_on_thresh){
    // Serial.println(cardOn);
    digitalWrite(12, LOW);
  } else {
    digitalWrite(12, HIGH);
  }

  //Serial.print("mV");
  Serial.println(map(avg, min, max, 0, 100));
    // Serial.print(min);
    // Serial.print(", ");
    // Serial.print(current);
    // Serial.print(", ");
    // Serial.print(max);
    // Serial.print("\n");
  /*************************************************************************************
   * Step 2.)
   * Keep running the same code as above to set up the sensitivity
   * (how many millivolts are output per Amp of current.
   * 
   * This time, use a known load current (measure this with a multimeter)
   * to give a constant output voltage. Adjust the sensitivity by turning the
   * gain potentiometer.
   * 
   * The sensitivity will be (known current)/(Vreading - Vref).
   *************************************************************************************/

    /*************************************************************************************
   * Step 3.)
   * Comment out the code used for the last two parts and uncomment the following code.
   * When you have performed the calibration steps above, make sure to change the 
   * global variables "sensitivity" and "Vref" to what you have set up.
   * 
   * This next line of code will print out the calculated current from these parameters.
   * The output is in mA
   *************************************************************************************/

  //Serial.print(current);
  //Serial.print("mA");

  // Reset the sensor value for the next reading
  sensorValue = 0;
  if (Serial.available()) {
    String input = Serial.readString();
    // Serial.print(input);
    if (input.indexOf("RELEASE") > -1) {
    //   // do something when the command "RELEASE" is received
      bool normal = LOW;
      digitalWrite(11, !normal);
      // Serial.print(!normal);
      delay(900);
      digitalWrite(11, normal);
      Serial.println("RELEASED");
    } else if (input.indexOf("?") > -1) {
      Serial.println("ARDUINO");      
    }
  }
}
