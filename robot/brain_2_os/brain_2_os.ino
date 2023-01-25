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
#define BUFFER_SIZE 64
char buffer[BUFFER_SIZE];
int bufferIndex = 0;

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

void print_current_vals(){
  for(int j=0;j<2;j++){
    unsigned long min = 10000;
    unsigned long max = 0;
    double avg = 0;
    Serial.flush();
    for (int j = 0; j < avgSamples; j++) {
      long current = analogRead(analogInPin);
      Serial.print(current);
      Serial.println(" ARDUINO");
      if(current < min){
        min = current;
      }  
      if(current > max){
        max = current;
      }
      avg += current;
      delay(2);
    }
    Serial.println("FIN ARDUINO");  
  }
}

void loop() {
  if (Serial.available() > 0) {
    char inChar = Serial.read();
    buffer[bufferIndex] = inChar;
    bufferIndex++;

    if (bufferIndex >= BUFFER_SIZE || inChar == '\n'  || inChar == '\r') {
      buffer[bufferIndex] = '\0';
      processData(buffer);
      bufferIndex = 0;
    }
  }
}

void processData(char* data) {
  String input = String(data);
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
    } else if (input.indexOf("CURRENT") > -1){
      print_current_vals();
    }
}