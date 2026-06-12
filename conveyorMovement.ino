// Pins for an L298N Driver
int enA = 9;  // Enable pin (Now acts as an ON/OFF switch)
int in1 = 8;  // Direction pin 1
int in2 = 7;  // Direction pin 2

void setup() {
  pinMode(enA, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    char key = Serial.read();

    // Move Forward
    if (key == 'w' || key == 'W') {
      digitalWrite(in1, HIGH);
      digitalWrite(in2, LOW);
      digitalWrite(enA, HIGH); // Turn motor ON
    } 
    // Move Backward
    else if (key == 's' || key == 'S') {
      digitalWrite(in1, LOW);
      digitalWrite(in2, HIGH);
      digitalWrite(enA, HIGH); // Turn motor ON
    }
    // Stop
    else if (key == 't' || key == 'T') { 
      digitalWrite(enA, LOW);  // Turn motor OFF
      digitalWrite(in1, LOW);  
      digitalWrite(in2, LOW);  
    }
  }
}