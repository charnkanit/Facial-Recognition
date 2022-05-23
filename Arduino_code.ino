#define maglock 8

int x;

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(1);
  pinMode(maglock, OUTPUT);
}

void loop() {
  while (!Serial.available());
  x = Serial.readString().toInt();
  Serial.print(String(x) + " => ");
  if(x == 9){
    digitalWrite(maglock, HIGH);
    Serial.println("Door open");
    delay(10000);
    digitalWrite(maglock, LOW);
    Serial.println("Door close");
    delay(1000);
  }
  else{
    digitalWrite(maglock, LOW);
    Serial.println("Door close");
  }
}
