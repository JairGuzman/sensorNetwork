#define trigPin1 3
#define echoPin1 2
#define trigPin2 5
#define echoPin2 4
#define trigPin3 7
#define echoPin3 6
#define trigPin4 9
#define echoPin4 8
#define trigPin5 11
#define echoPin5 10
#define trigPin6 13
#define echoPin6 12
long duration, distance, S1, S2, S3, S4, S5, S6;
void setup()
{
Serial.begin (9600);
  pinMode(trigPin1, OUTPUT);
  pinMode(echoPin1, INPUT);
  pinMode(trigPin2, OUTPUT);
  pinMode(echoPin2, INPUT);
  pinMode(trigPin3, OUTPUT);
  pinMode(echoPin3, INPUT);
  pinMode(trigPin4, OUTPUT);
  pinMode(echoPin4, INPUT);
  pinMode(trigPin5, OUTPUT);
  pinMode(echoPin5, INPUT);
  pinMode(trigPin6, OUTPUT);
  pinMode(echoPin6, INPUT);
}
void loop() {
  SonarSensor(trigPin1, echoPin1);
  S1 = distance;
  SonarSensor(trigPin2, echoPin2);
  S2 = distance;
  SonarSensor(trigPin3, echoPin3);
  S3 = distance;
  SonarSensor(trigPin4, echoPin4);
  S4 = distance;
  SonarSensor(trigPin5, echoPin5);
  S5 = distance;
  SonarSensor(trigPin6, echoPin6);
  S6 = distance;
  Serial.print(S1);
  Serial.print(",");
  Serial.print(S2);
  Serial.print(",");
  Serial.print(S3);
  Serial.print(",");
  Serial.print(S4);
  Serial.print(",");
  Serial.print(S5);
  Serial.print(",");
  Serial.print(S6);
  Serial.println(";");
  delay(8000);
}
void SonarSensor(int trigPin,int echoPin){
  //2 segundos
  digitalWrite(trigPin, LOW); //para generar un pulso limpio ponemos a LOW 4us
  delayMicroseconds(20); //tiempo que dura el pulso en HIGH
  digitalWrite(trigPin, HIGH); //generamos Trigger (disparo) de 10us
  delayMicroseconds(20); //mantenemos el pulso HIGH por 10us
  digitalWrite(trigPin, LOW); //generamos un pulso de 10us para indicar que ya no se enviara mas TRIGGER
  duration = pulseIn(echoPin, HIGH); //medimos el tiempo entre pulsos, en microsegundos
  distance = (duration/2) / 29.1; //convertimos a distancia, en cm
  if(distance > 100){ //si la distancia es mayor a 100cm
    distance = 0; //marcamos la distancia como 0
  }
}