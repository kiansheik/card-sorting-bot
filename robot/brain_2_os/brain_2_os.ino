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
#define BUFFER_SIZE 64*2
#define VACUUM_PIN 11
#define SOLENOID_PIN 12
char buffer[BUFFER_SIZE];
int bufferIndex = 0;

const int analogInPin = A0;

// Number of samples to average the reading over
// Change this to make the reading smoother... but beware of buffer overflows!
const long avgSamples = 100;
const int card_on_thresh = 1;
long sensorValue = 0;
long cardOn = 0;
long min = 100000;
bool vacuumOff = true;
float sensitivity = 100.0 / 500.0; //100mA per 500mV = 0.2
float Vref = 2500; // Output voltage with no current: ~ 2500mV or 2.5V

void setup() {
  // initialize serial communications at 9600 bps:
  Serial.begin(115200);
  pinMode(SOLENOID_PIN, OUTPUT);
  pinMode(VACUUM_PIN, OUTPUT);
  calibrate_vacuum();
}

void print_current_vals(){
  for(int i=0;i<1;i++){
      for (int j = 0; j < avgSamples; j++) {        
        Serial.print(((long)analogRead(analogInPin))- min);
        Serial.println(" ARDUINO");  
          delay(2);
        }
    Serial.println("FIN ARDUINO");  
  }
}

void calibrate_vacuum(){
  digitalWrite(VACUUM_PIN, LOW);
  int current = 0;
  for(int i=0;i<avgSamples*10;i++){
    current = analogRead(analogInPin);
    if (current < min){
      min = current;
    }
    delay(2);
  }      
  Serial.println("VACUUM_CALIBRATED");  
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
  // print_current_vals();
}

void processData(char* data) {
  String input = String(data);
    if (input.indexOf("RELEASE") > -1) {
      digitalWrite(SOLENOID_PIN, HIGH);
      delay(900);
      digitalWrite(SOLENOID_PIN, LOW);
      Serial.println("RELEASED");
    } else if (input.indexOf("?") > -1) {
      Serial.println("ARDUINO");      
    } else if (input.indexOf("CURRENT") > -1){
      print_current_vals();
    } else if (input.indexOf("VACUUM_ON") > -1){
      digitalWrite(VACUUM_PIN, HIGH);
      Serial.println("VACUUM_ON");  
      Serial.println("VACUUM_ON");  
    } else if (input.indexOf("VACUUM_OFF") > -1){
      digitalWrite(VACUUM_PIN, LOW);
      Serial.println("VACUUM_OFF");  
      Serial.println("VACUUM_OFF");  
    } else if (input.indexOf("CALIBRATE_VACUUM") > -1) {
        calibrate_vacuum();      
    }
}