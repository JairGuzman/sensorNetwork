const int s1 = A0;
const int s2 = A1;
const int s3 = A2;
const int s4 = A3;
const int s5 = A4;
const int s6 = A5;

void setup() {
  Serial.begin(9600);
}

void loop() {
  Serial.print(analogRead(s1));
  Serial.print(",");
  Serial.print(analogRead(s2));
  Serial.print(",");
  Serial.print(analogRead(s3));
  Serial.print(",");
  Serial.print(analogRead(s4));
  Serial.print(",");
  Serial.print(analogRead(s5));
  Serial.print(",");
  Serial.print(analogRead(s6));
  Serial.println(";");

  delay(10000);

  
}