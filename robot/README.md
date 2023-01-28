# GRBL Controller

`control.py` provides a class `GRBL_Controller` that interfaces with the GRBL CNC controller and an Arduino board to control a vacuum pump and camera on an x,y,z gantry. arduino controls the vacuum motor relay, vacuum release solenoid relay, and reading the values of the pressure sensor (or current sensor) using an SVM to auto-calibrate your own thresholds to have the freedom to use a sensor that works for you.

## Dependencies
- glob
- time
- random
- cv2
- numpy
- serial
### Local Dependencies
- aruco
- pressure_svm
- mtg_card_detector

## Usage
1. Initialize an instance of the `GRBL_Controller` class by providing it with the serial objects for the GRBL controller and the Arduino board, as well as a capture object for the camera.
2. Call the `init()` method on the instance to initialize the GRBL controller and the vacuum pump.
3. Call the `deinit()` method on the instance to close the GRBL controller and the vacuum pump.
4. Use the other methods provided by the class to control the stepper motor and the vacuum pump.
5. Use the `calibrations` dictionary to access the calibration data of the system, which includes the stepper motor and aruco marker positions, and aruco IDs for each stack in your card-stack matrix.

# GRBL_Controller

The GRBL_Controller class is a python wrapper for controlling a [GRBL](https://github.com/gnea/grbl) device using the [Serial library](https://pyserial.readthedocs.io/en/latest/pyserial.html).

## Methods

### __init__(self, grbl, arduino, cap)
Initializes the class with a GRBL serial connection, an arduino serial connection, and a cv2 capture object.

### init(self)
Initializes the GRBL device and calibrates the vacuum.

### deinit(self)
Deinitializes the GRBL device and returns it to home.

### wait_till_on(self, ser)
Waits until the serial device is on.

### check_status(self, ser)
Returns the `?` command status of the serial device.

### wait_till_ready(self)
Waits until the GRBL device is ready to receive commands.

### step(self, steps, axis, direction)
Sends a step command to the GRBL device.

### reset_zero(self)
Resets zero position to current position.

### go_to(self, x=None, y=None, z=None, relative=False)
Sends a go-to command to the GRBL device. relative=True is equal to step and maybe they will be merged

### release(self)
Releases vacuum.

### home(self)
Homes the GRBL device.

### calibrate_vacuum(self)
Calibrates the vacuum sensor value, must be called when vacuum is off.

### detect_card(self, (row, col))
Attempts to detect a card at the given stack id defined in the calibrations dict
