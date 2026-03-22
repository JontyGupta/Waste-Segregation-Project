/*
 * =================================================================
 * Waste Classifier Arduino Firmware
 * =================================================================
 *
 * Controls 10 servo motors, one per waste bin.
 * Receives serial commands from the Python controller.
 * 
 * Protocol:
 *   ROUTE:<bin_id>         → Open gate, wait, close gate
 *   OPEN:<bin_id>          → Open gate (stay open)
 *   CLOSE:<bin_id>         → close gate
 *   STATUS                 → Report all servo positions
 *   RESET                  → close all gates
 *   PING                   → Reply PONG
 *   LED: <bin>:<r>:<g>:<b> → Set bin LED color (if RGB LEDS)
 *
 * Hardware Wiring:
 *   Servo 0 (Biodegradable)       → Pin 2
 *   Servo 1 (Recyclable)          → Pin 3
 *   Servo 2 (Non-Recyclable)      → Pin 4
 *   Servo 3 (Medical)             → Pin 5
 *   Servo 4 (E-Waste)             → Pin 6
 *   Servo 5 (Hazardous)           → Pin 7
 *   Servo 6 (Textile)             → Pin 8
 *   Servo 7 (Construction)        → Pin 9
 *   Servo 8 (Sanitary)            → Pin 10
 *   Servo 9 (Other)               → Pin 11
 *  
 * Board: Arduino Mega 2560 (recommended for 10 servos) 
 *        Arduino Uno works with external servo driver (PCA9685)
 * =================================================================
 */

#include <Servo.h>

// ======================= CONFIGURATION =========================

#define NUM_BINS        10
#define BAUD_RATE       9600
#define SERIAL_TIMEOUT  100  // ms
#define BUFFER_SIZE     64

// Servo pins - must match Python bin_config.py
const int SERVO_PINS[NUM_BINS] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

// Servo angles
const int OPEN_ANGLE = 90;
const int CLOSE_ANGLE = 0;

// Hold times per bin (ms) how long the gate stays open
const int HOLD_TIMES[NUM_BINS] = {
    3000, // Bin 0: Biodegradable
    3000, // Bin 1: Recyclable
    3000, // Bin 2: Non-Recyclable
    4000, // Bin 3: Medical (longer for safety)
    3000, // Bin 4: E-Waste
    4000, // Bin 5: Hazardous (longer for safety)
    3000, // Bin 6: Textile
    3500, // Bin 7: Construction
    3000, // Bin 8: Sanitary
    3000  // Bin 9: Other
};

// Bin labels for status reporting
const char* BIN_LABELS[NUM_BINS] = {
    "Biodegradable", "Recyclable", "NonRecyclable", 
    "Medical", "Ewaste", "Hazardous",
    "Textile", "Construction", "Sanitary", "Other"
};

// ======================= GLOBALS =========================

Servo servos[NUM_BINS];
bool gateOpen[NUM_BINS];
char inputBuffer[BUFFER_SIZE];
int bufferIndex = 0;

// ======================= SETUP =========================

void setup() {
    Serial.begin(BAUD_RATE);
    Serial.setTimeout(SERIAL_TIMEOUT);

    // Attach all servos and close all gates
    for (int i = 0; i < NUM_BINS; i++) {
        servos[i].attach(SERVO_PINS[i]);
        servos[i].write(CLOSE_ANGLE);
        gateOpen[i] = false;
    }

    delay(500);
    Serial.println("READY: WasteClassifier");
}

// ======================= MAIN LOOP =========================

void loop() {
    // Read serial data character by character 
    while (Serial.available() > 0) {
        char c = Serial.read();
        
        if (c == '\n' || c = '\r') {
            if (bufferIndex > 0) {
                inputBuffer[bufferIndex] = '\0';
                processCommand(inputBuffer);
                bufferIndex = 0;
            } 
        } else if (bufferIndex < BUFFER_SIZE - 1) {
            inputBuffer[bufferIndex++] = c;
        }
    }
}

// ======================= COMMAND PROCESSOR =========================

void processCommand(const char* cmd) {
    // Parse command and payload
    String command = String(cmd);
    String action = "";
    String payload = "";

    int colonIdx = command.indexOf(':');
    if (colonIdx >= 0) {
        action = command.substring(0, colonIdx);
        payload = command.substring(colonIdx + 1);
    } else {
        action = command;
    }

    action.trim();
    action.toUpperCase();

    // ----------- PING -----------
    if (action == "PING") {
        Serial.println("PONG");
        return;
    }

    // ----------- STATUS -----------
    if (action == "STATUS") {
        sendStatus();
        return;
    }

    // ----------- RESET -----------
    if (action == "RESET") {
        resetAll();
        Serial.println("ACK:RESET");
        return;
    }
    
    // ----------- ROUTE -----------
    if (action == "ROUTE") {
        int binId = payload.toInt(); 
        if (isValidBin(binId)) { 
            routeWaste(binId); 
            Serial.print("ACK:ROUTE:"); 
            Serial.println(binId); 
        } else {
            Serial.print("ERR: Invalid bin ID ");
            Serial.println(binId); 
        }
        return;
    }

    // ----------- OPEN -----------
    if (action == "OPEN") {
        int binId = payload.toInt(); 
        if (isValidBin(binId)) { 
            openGate(binId); 
            Serial.print("ACK:OPEN:"); 
            Serial.println(binId); 
        } else {
            Serial.print("ERR: Invalid bin ID ");
            Serial.println(binId);
        }
        return;
    }
   
    // ----------- CLOSE -----------
    if (action == "CLOSE") {
        int binId = payload.toInt();
        if (isValidBin(binId)) {
            closeGate(binId);
            Serial.print("ACK:CLOSE:");
            Serial.println(binId);
        } else {
            Serial.print("ERR: Invalid bin ID ");
            Serial.println(binId);
        }
        return;
    }
    
    // ----------- LED -----------
    if (action == "LED") {
        // payload: <bin_id>:<r>:<g>:<b>
        Serial.print("ACK:LED:");
        Serial.println(payload);
        // TODO: Implement RGB LED control based on your hardware
        return;
    }
    
    // Unknown command
    Serial.print("ERR: Unknown command '");
    Serial.print(action);
    Serial.println("'");
}
    
// ======================= MOTOR CONTROL =========================

void routewaste(int binId) {
    // close any currently open gate first 
    for (int i = 0; i < NUM_BINS; i++) { 
        if (gateOpen[i] && i I= binId) { 
            closeGate(i); 
        }
    }

    // Open target bin gate 
    openGate(binId);
    
    // Wait for waste to fall into bin 
    delay(HOLD_TIMES[binId]);
    
    // Close gate 
    closeGate(binId);
}

void openGate(int binId) { 
    servos[binId].write(OPEN_ANGLE); 
    gateOpen[binId] = true;
}   

void closeGate(int binId) { 
    servos[binId].write(CLOSE_ANGLE); 
    gateOpen[binId] = false;
}

void resetAll() {
    for (int i = 0; i < NUM_BINS; i++) { 
        closeGate(i); 
        delay(100); // Stagger to avoid power surge
    } 
}

bool isValidBin(int binId) {
    return binId >= 0 && binId < NUM BINS;
}

// ======================= STATUS REPORT =========================

void sendstatus() {
    // Send a simple status report
    Serial.print("STATUS:{\"bins\":[");
    for (int i = 0; i < MUM_BINS; i++) {
        if (i > 0) serial.print(",");
        Serial.print("{\"id\":");
        Serial.print(i);
        Serial.print(",\"label\":\"");
        Serial.print(BIN_LABELS[i]);
        Serial.print("\",\"pin\":");
        Serial.print(SERVO_PINS[i]);
        Serial.print(",\"open\":");
        Serial.print(gateopen[i] ? "true": "false"); 
        Serial.print(",\"angle\":");
        Serial.print(servos[i].read());
        Serial.print("}");
    }
    Serial.println("]}");
}
