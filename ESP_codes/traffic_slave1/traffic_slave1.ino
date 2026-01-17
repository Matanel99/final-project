#include <WiFi.h>
#include <esp_now.h>

#define RED_PIN    13
#define YELLOW_PIN 12
#define GREEN_PIN  14

enum MsgType    : uint8_t { MSG_JOYSTICK = 1, MSG_TRAFFIC = 2 };
enum TrafficCmd : uint8_t { TL_NEXT = 1 };
enum TargetId   : uint8_t { TGT_A = 1, TGT_B = 2, TGT_ALL = 255 };

typedef struct __attribute__((packed)) {
  uint8_t type;      // MSG_TRAFFIC
  uint8_t cmd;       // TL_NEXT
  uint8_t target_id; // TGT_A / TGT_B / TGT_ALL
} msg_traffic_t;

const uint8_t MY_ID = TGT_A; // ברמזור B שנה ל-TGT_B
volatile int stateTL = 0;    // 0=RED, 1=GREEN, 2=YELLOW

void setLights(int s){
  digitalWrite(RED_PIN,    s == 0 ? HIGH : LOW);
  digitalWrite(GREEN_PIN,  s == 1 ? HIGH : LOW);
  digitalWrite(YELLOW_PIN, s == 2 ? HIGH : LOW);
}

void onRecv(const esp_now_recv_info *info, const uint8_t *data, int len){
  if (len < (int)sizeof(msg_traffic_t)) return;
  const msg_traffic_t *m = (const msg_traffic_t*)data;
  if (m->type != MSG_TRAFFIC) return;

  if (m->target_id == MY_ID || m->target_id == TGT_ALL) {
    if (m->cmd == TL_NEXT) {
      stateTL = (stateTL + 1) % 3;
      setLights(stateTL);
    }
  }
}

void setup(){
  pinMode(RED_PIN, OUTPUT);
  pinMode(YELLOW_PIN, OUTPUT);
  pinMode(GREEN_PIN, OUTPUT);
  setLights(stateTL);

  Serial.begin(115200);
  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) { while(1) delay(1000); }
  esp_now_register_recv_cb(onRecv);
}

void loop(){ delay(1); }
