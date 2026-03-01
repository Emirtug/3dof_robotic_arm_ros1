/*
 * OpenCM9.04 Robot Arm Controller - WITH SAFETY FEATURES
 * Receives joint positions via Serial from Python MQTT controller
 * Controls 3x AX-12A Dynamixel servos
 * 
 * SAFETY FEATURES:
 * - Watchdog timer (auto-safe on communication loss)
 * - Voltage monitoring (prevents damage from low/high voltage)
 * - Temperature monitoring (prevents overheating)
 * - Torque limiting (prevents servo strain)
 * - Emergency stop (E-STOP) with torque disable
 * - Soft start (gradual movement on power-up)
 * - Heartbeat monitoring
 * 
 * Protocol:
 *   Joints:    "J:j1,j2,j3\n"  (radians)
 *   Commands:  "C:home\n", "C:safe\n", "C:stop\n", "C:start\n"
 *              "C:estop\n", "C:reset\n", "C:relax\n"
 *   Heartbeat: "H:\n"
 * 
 * Wiring (OpenCM9.04 Expansion Board):
 *   - AX-12A connected to TTL port
 *   - Servo IDs: 1, 2, 3
 *   - Optional: E-STOP button on pin 23 (active LOW)
 */

#include <Dynamixel2Arduino.h>

// ============ DYNAMIXEL SETTINGS ============
#define DXL_SERIAL   Serial1
#define DEBUG_SERIAL Serial

const uint8_t DXL_DIR_PIN = 28;  // OpenCM9.04 direction pin

// Servo IDs (change if different)
const uint8_t SERVO_ID_1 = 1;  // Base (Joint1)
const uint8_t SERVO_ID_2 = 2;  // Shoulder (Joint2)
const uint8_t SERVO_ID_3 = 3;  // Elbow (Joint3)

// Protocol version for AX-12A
const float DXL_PROTOCOL_VERSION = 1.0;

// ============ SAFETY SETTINGS ============
// Watchdog timeout (ms) - go to safe if no data received
const unsigned long WATCHDOG_TIMEOUT = 1500;

// Heartbeat timeout (ms) - expect heartbeat this often
const unsigned long HEARTBEAT_TIMEOUT = 2000;

// Voltage limits (AX-12A: 9-12V nominal)
const float VOLTAGE_MIN = 9.5;   // Volts - below this is dangerous
const float VOLTAGE_MAX = 13.0;  // Volts - above this is dangerous
const unsigned long VOLTAGE_CHECK_INTERVAL = 1000;  // Check every 1s

// Temperature limit (AX-12A max ~70°C)
const uint8_t TEMP_MAX = 65;  // Celsius - warning threshold
const unsigned long TEMP_CHECK_INTERVAL = 2000;  // Check every 2s

// Torque limit (0-1023, lower = safer)
const int TORQUE_LIMIT = 512;  // 50% max torque

// Movement speed (slower = safer)
const int SERVO_SPEED_NORMAL = 100;   // Normal operation
const int SERVO_SPEED_SLOW = 30;      // Soft start / safe moves

// E-STOP button pin (optional hardware)
const uint8_t ESTOP_PIN = 23;  // Active LOW

// Status LED (built-in on OpenCM9.04)
const uint8_t LED_PIN = 14;  // User LED

// ============ JOINT LIMITS (radians) ============
const float JOINT1_MIN = -3.14159;
const float JOINT1_MAX = 3.14159;
const float JOINT2_MIN = 0.0;
const float JOINT2_MAX = 1.5708;
const float JOINT3_MIN = 0.0;
const float JOINT3_MAX = 1.5708;

// ============ AX-12A SETTINGS ============
const int AX12_MIN = 0;
const int AX12_MAX = 1023;
const int AX12_CENTER = 512;

// ============ HOME/SAFE POSITIONS (radians) ============
const float HOME_J1 = 0.0;
const float HOME_J2 = 0.5;
const float HOME_J3 = 0.5;

const float SAFE_J1 = 0.0;
const float SAFE_J2 = 0.3;
const float SAFE_J3 = 0.3;

// ============ GLOBAL VARIABLES ============
Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);

float target_j1 = HOME_J1;
float target_j2 = HOME_J2;
float target_j3 = HOME_J3;

// Timing
unsigned long last_data_time = 0;
unsigned long last_heartbeat_time = 0;
unsigned long last_voltage_check = 0;
unsigned long last_temp_check = 0;

// State flags
bool control_enabled = false;       // Start disabled until handshake
bool watchdog_triggered = false;
bool emergency_stop = false;
bool servos_initialized = false;
bool torque_enabled = false;

// Error counters
int voltage_error_count = 0;
int temp_warning_count = 0;

String input_buffer = "";

// ============ FUNCTION PROTOTYPES ============
void processSerialData();
void parseJointCommand(String data);
void parseControlCommand(String data);
void moveServos();
void moveServosSlowly();
void goHome();
void goSafe();
void initServos();
void enableTorque();
void disableTorque();
void triggerEmergencyStop(const char* reason);
void resetEmergencyStop();
void checkVoltage();
void checkTemperature();
void checkHardwareESTOP();
void blinkLED(int times, int delayMs);
void statusReport();

// ============ SETUP ============
void setup() {
  // Debug serial (USB)
  DEBUG_SERIAL.begin(115200);
  while (!DEBUG_SERIAL && millis() < 3000);
  
  DEBUG_SERIAL.println("============================================");
  DEBUG_SERIAL.println("  OpenCM9.04 Robot Controller");
  DEBUG_SERIAL.println("  WITH SAFETY FEATURES");
  DEBUG_SERIAL.println("============================================");
  
  // Setup pins
  pinMode(LED_PIN, OUTPUT);
  pinMode(ESTOP_PIN, INPUT_PULLUP);  // E-STOP with pull-up
  
  // Blink LED to indicate startup
  blinkLED(3, 200);
  
  // Dynamixel serial
  dxl.begin(1000000);
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);
  
  // Initialize servos (but don't enable torque yet)
  initServos();
  
  DEBUG_SERIAL.println("[OK] Waiting for connection...");
  DEBUG_SERIAL.println("     Send 'C:start' to begin control");
  DEBUG_SERIAL.println("============================================");
  
  last_data_time = millis();
  last_heartbeat_time = millis();
}

// ============ MAIN LOOP ============
void loop() {
  // Check hardware E-STOP button
  checkHardwareESTOP();
  
  // If emergency stopped, just blink LED and wait
  if (emergency_stop) {
    static unsigned long last_blink = 0;
    if (millis() - last_blink > 200) {
      digitalWrite(LED_PIN, !digitalRead(LED_PIN));
      last_blink = millis();
    }
    processSerialData();  // Still process commands for reset
    delay(10);
    return;
  }
  
  // Process incoming serial data
  processSerialData();
  
  // Periodic safety checks
  if (millis() - last_voltage_check > VOLTAGE_CHECK_INTERVAL) {
    checkVoltage();
    last_voltage_check = millis();
  }
  
  if (millis() - last_temp_check > TEMP_CHECK_INTERVAL) {
    checkTemperature();
    last_temp_check = millis();
  }
  
  // Watchdog - check for communication timeout
  if (control_enabled && torque_enabled) {
    // Check heartbeat timeout
    if (millis() - last_heartbeat_time > HEARTBEAT_TIMEOUT) {
      if (!watchdog_triggered) {
        DEBUG_SERIAL.println("[WARN] Heartbeat timeout - going to safe");
        goSafe();
        watchdog_triggered = true;
      }
    }
    
    // Check data timeout
    if (millis() - last_data_time > WATCHDOG_TIMEOUT) {
      if (!watchdog_triggered) {
        DEBUG_SERIAL.println("[WARN] Watchdog: No data - going to safe");
        goSafe();
        watchdog_triggered = true;
      }
    }
  }
  
  // LED status indicator
  if (control_enabled && !watchdog_triggered) {
    digitalWrite(LED_PIN, HIGH);  // Solid = running
  } else if (watchdog_triggered) {
    // Slow blink = watchdog triggered
    static unsigned long last_blink = 0;
    if (millis() - last_blink > 500) {
      digitalWrite(LED_PIN, !digitalRead(LED_PIN));
      last_blink = millis();
    }
  } else {
    digitalWrite(LED_PIN, LOW);  // Off = not running
  }
  
  delay(10);
}

// ============ SERIAL PROCESSING ============
void processSerialData() {
  while (DEBUG_SERIAL.available() > 0) {
    char c = DEBUG_SERIAL.read();
    
    if (c == '\n' || c == '\r') {
      if (input_buffer.length() > 0) {
        if (input_buffer.startsWith("J:")) {
          parseJointCommand(input_buffer.substring(2));
        } else if (input_buffer.startsWith("C:")) {
          parseControlCommand(input_buffer.substring(2));
        } else if (input_buffer.startsWith("H:")) {
          // Heartbeat received
          last_heartbeat_time = millis();
        } else if (input_buffer == "?") {
          statusReport();
        } else {
          DEBUG_SERIAL.println("[ERR] Unknown command");
        }
        input_buffer = "";
      }
    } else {
      input_buffer += c;
      if (input_buffer.length() > 64) {
        input_buffer = "";
        DEBUG_SERIAL.println("[ERR] Buffer overflow");
      }
    }
  }
}

void parseJointCommand(String data) {
  if (emergency_stop) {
    DEBUG_SERIAL.println("[ERR] E-STOP active, ignoring command");
    return;
  }
  
  int comma1 = data.indexOf(',');
  int comma2 = data.indexOf(',', comma1 + 1);
  
  if (comma1 == -1 || comma2 == -1) {
    DEBUG_SERIAL.println("[ERR] Invalid joint format");
    return;
  }
  
  float j1 = data.substring(0, comma1).toFloat();
  float j2 = data.substring(comma1 + 1, comma2).toFloat();
  float j3 = data.substring(comma2 + 1).toFloat();
  
  // Clamp to limits
  target_j1 = constrain(j1, JOINT1_MIN, JOINT1_MAX);
  target_j2 = constrain(j2, JOINT2_MIN, JOINT2_MAX);
  target_j3 = constrain(j3, JOINT3_MIN, JOINT3_MAX);
  
  // Update timing
  last_data_time = millis();
  last_heartbeat_time = millis();  // Joint data counts as heartbeat
  watchdog_triggered = false;
  
  // Move servos if enabled
  if (control_enabled && torque_enabled) {
    moveServos();
  }
}

void parseControlCommand(String data) {
  data.trim();
  data.toLowerCase();
  
  if (data == "home") {
    DEBUG_SERIAL.println("[CMD] Going to HOME");
    if (!emergency_stop) {
      goHome();
    }
    
  } else if (data == "safe") {
    DEBUG_SERIAL.println("[CMD] Going to SAFE");
    goSafe();
    
  } else if (data == "stop") {
    DEBUG_SERIAL.println("[CMD] Control STOPPED");
    control_enabled = false;
    
  } else if (data == "start") {
    if (emergency_stop) {
      DEBUG_SERIAL.println("[ERR] Cannot start - E-STOP active. Send 'C:reset' first");
    } else {
      DEBUG_SERIAL.println("[CMD] Control STARTED");
      control_enabled = true;
      watchdog_triggered = false;
      last_data_time = millis();
      last_heartbeat_time = millis();
      
      // Enable torque and do soft start
      if (!torque_enabled) {
        enableTorque();
        goHome();  // Soft start to home
      }
    }
    
  } else if (data == "estop") {
    triggerEmergencyStop("Remote E-STOP command");
    
  } else if (data == "reset") {
    resetEmergencyStop();
    
  } else if (data == "relax") {
    DEBUG_SERIAL.println("[CMD] Relaxing servos (torque off)");
    disableTorque();
    
  } else if (data == "status") {
    statusReport();
    
  } else {
    DEBUG_SERIAL.print("[ERR] Unknown command: ");
    DEBUG_SERIAL.println(data);
  }
}

// ============ SERVO CONTROL ============
void initServos() {
  DEBUG_SERIAL.println("[INIT] Scanning servos...");
  
  uint8_t ids[] = {SERVO_ID_1, SERVO_ID_2, SERVO_ID_3};
  const char* names[] = {"Base", "Shoulder", "Elbow"};
  int found_count = 0;
  
  for (int i = 0; i < 3; i++) {
    if (dxl.ping(ids[i])) {
      DEBUG_SERIAL.print("  [OK] ID ");
      DEBUG_SERIAL.print(ids[i]);
      DEBUG_SERIAL.print(" (");
      DEBUG_SERIAL.print(names[i]);
      DEBUG_SERIAL.println(")");
      
      // Configure servo
      dxl.torqueOff(ids[i]);
      dxl.setOperatingMode(ids[i], OP_POSITION);
      
      // Set torque limit for safety
      dxl.writeControlTableItem(TORQUE_LIMIT, ids[i], TORQUE_LIMIT);
      
      // Set slow speed initially
      dxl.writeControlTableItem(MOVING_SPEED, ids[i], SERVO_SPEED_SLOW);
      
      found_count++;
    } else {
      DEBUG_SERIAL.print("  [FAIL] ID ");
      DEBUG_SERIAL.print(ids[i]);
      DEBUG_SERIAL.print(" (");
      DEBUG_SERIAL.print(names[i]);
      DEBUG_SERIAL.println(") NOT found!");
    }
  }
  
  servos_initialized = (found_count == 3);
  
  if (!servos_initialized) {
    DEBUG_SERIAL.println("[WARN] Not all servos found! Check connections.");
  }
}

void enableTorque() {
  if (!servos_initialized) {
    DEBUG_SERIAL.println("[ERR] Cannot enable torque - servos not initialized");
    return;
  }
  
  DEBUG_SERIAL.println("[SAFETY] Enabling torque with slow speed...");
  
  uint8_t ids[] = {SERVO_ID_1, SERVO_ID_2, SERVO_ID_3};
  for (int i = 0; i < 3; i++) {
    dxl.writeControlTableItem(MOVING_SPEED, ids[i], SERVO_SPEED_SLOW);
    dxl.torqueOn(ids[i]);
  }
  
  torque_enabled = true;
  delay(100);
  
  // Switch to normal speed after a moment
  delay(500);
  for (int i = 0; i < 3; i++) {
    dxl.writeControlTableItem(MOVING_SPEED, ids[i], SERVO_SPEED_NORMAL);
  }
  
  DEBUG_SERIAL.println("[OK] Torque enabled");
}

void disableTorque() {
  uint8_t ids[] = {SERVO_ID_1, SERVO_ID_2, SERVO_ID_3};
  for (int i = 0; i < 3; i++) {
    dxl.torqueOff(ids[i]);
  }
  torque_enabled = false;
  control_enabled = false;
  DEBUG_SERIAL.println("[OK] Torque disabled - servos relaxed");
}

void moveServos() {
  if (!torque_enabled) return;
  
  // Convert radians to AX-12A positions
  int pos1 = AX12_CENTER + (int)(target_j1 / 3.14159 * (AX12_MAX - AX12_CENTER));
  pos1 = constrain(pos1, AX12_MIN, AX12_MAX);
  
  int pos2 = AX12_CENTER + (int)(target_j2 / 1.5708 * 170);
  pos2 = constrain(pos2, AX12_MIN, AX12_MAX);
  
  int pos3 = AX12_CENTER + (int)(target_j3 / 1.5708 * 170);
  pos3 = constrain(pos3, AX12_MIN, AX12_MAX);
  
  dxl.setGoalPosition(SERVO_ID_1, pos1);
  dxl.setGoalPosition(SERVO_ID_2, pos2);
  dxl.setGoalPosition(SERVO_ID_3, pos3);
}

void moveServosSlowly() {
  // Temporarily set slow speed
  uint8_t ids[] = {SERVO_ID_1, SERVO_ID_2, SERVO_ID_3};
  for (int i = 0; i < 3; i++) {
    dxl.writeControlTableItem(MOVING_SPEED, ids[i], SERVO_SPEED_SLOW);
  }
  
  moveServos();
  delay(500);  // Wait for movement
  
  // Restore normal speed if not in safe mode
  if (control_enabled && !watchdog_triggered) {
    for (int i = 0; i < 3; i++) {
      dxl.writeControlTableItem(MOVING_SPEED, ids[i], SERVO_SPEED_NORMAL);
    }
  }
}

void goHome() {
  target_j1 = HOME_J1;
  target_j2 = HOME_J2;
  target_j3 = HOME_J3;
  
  if (!torque_enabled) {
    enableTorque();
  }
  
  moveServosSlowly();
  
  control_enabled = true;
  watchdog_triggered = false;
  last_data_time = millis();
  last_heartbeat_time = millis();
  
  DEBUG_SERIAL.println("[OK] At HOME position");
}

void goSafe() {
  target_j1 = SAFE_J1;
  target_j2 = SAFE_J2;
  target_j3 = SAFE_J3;
  
  // Use slow speed for safe movement
  uint8_t ids[] = {SERVO_ID_1, SERVO_ID_2, SERVO_ID_3};
  for (int i = 0; i < 3; i++) {
    dxl.writeControlTableItem(MOVING_SPEED, ids[i], SERVO_SPEED_SLOW);
  }
  
  moveServos();
  
  DEBUG_SERIAL.println("[OK] Moving to SAFE position");
}

// ============ SAFETY FUNCTIONS ============
void triggerEmergencyStop(const char* reason) {
  if (emergency_stop) return;
  
  emergency_stop = true;
  control_enabled = false;
  
  DEBUG_SERIAL.println("==========================================");
  DEBUG_SERIAL.println("[ESTOP] !!! EMERGENCY STOP !!!");
  DEBUG_SERIAL.print("[ESTOP] Reason: ");
  DEBUG_SERIAL.println(reason);
  DEBUG_SERIAL.println("==========================================");
  
  // Immediately disable torque for safety
  disableTorque();
  
  DEBUG_SERIAL.println("[ESTOP] Send 'C:reset' to resume");
}

void resetEmergencyStop() {
  if (!emergency_stop) {
    DEBUG_SERIAL.println("[INFO] E-STOP not active");
    return;
  }
  
  // Check if hardware E-STOP is released
  if (digitalRead(ESTOP_PIN) == LOW) {
    DEBUG_SERIAL.println("[ERR] Release hardware E-STOP button first!");
    return;
  }
  
  DEBUG_SERIAL.println("[OK] E-STOP reset");
  emergency_stop = false;
  voltage_error_count = 0;
  temp_warning_count = 0;
  
  // Don't auto-enable - wait for explicit start command
  DEBUG_SERIAL.println("[INFO] Send 'C:start' to resume control");
}

void checkVoltage() {
  if (!servos_initialized) return;
  
  // Read voltage from first servo (all should be same)
  float voltage = dxl.readControlTableItem(PRESENT_VOLTAGE, SERVO_ID_1) / 10.0;
  
  if (voltage < VOLTAGE_MIN) {
    voltage_error_count++;
    DEBUG_SERIAL.print("[ERR] LOW VOLTAGE: ");
    DEBUG_SERIAL.print(voltage);
    DEBUG_SERIAL.println("V");
    
    if (voltage_error_count >= 3) {
      triggerEmergencyStop("Low voltage detected");
    }
  } else if (voltage > VOLTAGE_MAX) {
    voltage_error_count++;
    DEBUG_SERIAL.print("[ERR] HIGH VOLTAGE: ");
    DEBUG_SERIAL.print(voltage);
    DEBUG_SERIAL.println("V");
    
    if (voltage_error_count >= 3) {
      triggerEmergencyStop("High voltage detected");
    }
  } else {
    voltage_error_count = 0;  // Reset counter on good reading
  }
}

void checkTemperature() {
  if (!servos_initialized) return;
  
  uint8_t ids[] = {SERVO_ID_1, SERVO_ID_2, SERVO_ID_3};
  
  for (int i = 0; i < 3; i++) {
    int temp = dxl.readControlTableItem(PRESENT_TEMPERATURE, ids[i]);
    
    if (temp > TEMP_MAX) {
      temp_warning_count++;
      DEBUG_SERIAL.print("[WARN] OVERHEAT Servo ");
      DEBUG_SERIAL.print(ids[i]);
      DEBUG_SERIAL.print(": ");
      DEBUG_SERIAL.print(temp);
      DEBUG_SERIAL.println("°C");
      
      if (temp_warning_count >= 5) {
        triggerEmergencyStop("Servo overheating");
      }
    }
  }
}

void checkHardwareESTOP() {
  // E-STOP button is active LOW
  if (digitalRead(ESTOP_PIN) == LOW) {
    if (!emergency_stop) {
      triggerEmergencyStop("Hardware E-STOP pressed");
    }
  }
}

void blinkLED(int times, int delayMs) {
  for (int i = 0; i < times; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(delayMs);
    digitalWrite(LED_PIN, LOW);
    delay(delayMs);
  }
}

void statusReport() {
  DEBUG_SERIAL.println("========== STATUS ==========");
  DEBUG_SERIAL.print("  Control: ");
  DEBUG_SERIAL.println(control_enabled ? "ENABLED" : "DISABLED");
  DEBUG_SERIAL.print("  Torque: ");
  DEBUG_SERIAL.println(torque_enabled ? "ON" : "OFF");
  DEBUG_SERIAL.print("  E-STOP: ");
  DEBUG_SERIAL.println(emergency_stop ? "ACTIVE" : "Clear");
  DEBUG_SERIAL.print("  Watchdog: ");
  DEBUG_SERIAL.println(watchdog_triggered ? "TRIGGERED" : "OK");
  
  if (servos_initialized) {
    float voltage = dxl.readControlTableItem(PRESENT_VOLTAGE, SERVO_ID_1) / 10.0;
    DEBUG_SERIAL.print("  Voltage: ");
    DEBUG_SERIAL.print(voltage);
    DEBUG_SERIAL.println("V");
    
    DEBUG_SERIAL.print("  Temps: ");
    uint8_t ids[] = {SERVO_ID_1, SERVO_ID_2, SERVO_ID_3};
    for (int i = 0; i < 3; i++) {
      int temp = dxl.readControlTableItem(PRESENT_TEMPERATURE, ids[i]);
      DEBUG_SERIAL.print(temp);
      DEBUG_SERIAL.print("°C ");
    }
    DEBUG_SERIAL.println();
  }
  
  DEBUG_SERIAL.println("============================");
}

