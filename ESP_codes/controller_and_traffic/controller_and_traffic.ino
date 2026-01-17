// ========================= Master: ESP-NOW TX (Car + Two Traffic Lights) =========================
#include <WiFi.h>
#include <esp_now.h>

// -------------------- Pins --------------------
#define BTN_A_NEXT  13
#define BTN_B_NEXT  12   // avoid pressing during boot
#define BTN_BOTH    14

#define JOY_X       34   // ADC-capable
#define JOY_Y       35
#define JOY_SW      19

// -------------------- Peers MAC --------------------
uint8_t CAR_MAC[]      = {0x78, 0x42, 0x1C, 0x67, 0xC1, 0xB4};
uint8_t RAMZOR_A_MAC[] = {0xF4, 0x65, 0x0B, 0x45, 0xA3, 0xDC};
uint8_t RAMZOR_B_MAC[] = {0x78, 0x1C, 0x3C, 0xB9, 0x87, 0xB0};

// -------------------- Message types --------------------
enum MsgType     : uint8_t { MSG_JOYSTICK = 1, MSG_TRAFFIC = 2 };
enum TrafficCmd  : uint8_t { TL_NEXT = 1 };
enum TargetId    : uint8_t { TGT_A = 1, TGT_B = 2, TGT_ALL = 255 };

// Joystick message
typedef struct __attribute__((packed)) {
  uint8_t  type;    // MSG_JOYSTICK
  int16_t  joy_x;   // 0..4095
  int16_t  joy_y;   // 0..4095
  int8_t   joy_sw;  // 1=released, 0=pressed
} msg_joystick_t;

// Traffic-light command WITH TARGET
typedef struct __attribute__((packed)) {
  uint8_t type;      // MSG_TRAFFIC
  uint8_t cmd;       // TL_NEXT
  uint8_t target_id; // TGT_A / TGT_B / TGT_ALL
} msg_traffic_t;

union TxPayload {
  msg_joystick_t joy;
  msg_traffic_t  tl;
};
TxPayload tx;

// -------------------- Timing / Debounce --------------------
static const uint32_t NOW_PERIOD_MS = 30;
static const uint32_t DEBOUNCE_MS   = 25;

uint32_t lastNowMs = 0;

struct BtnState { uint8_t pin; int last, stable; uint32_t tchg; };
BtnState btnA   = { BTN_A_NEXT, HIGH, HIGH, 0 };
BtnState btnB   = { BTN_B_NEXT, HIGH, HIGH, 0 };
BtnState btnAll = { BTN_BOTH,   HIGH, HIGH, 0 };

// -------------------- ESP-NOW helpers --------------------
void onNowSend(const uint8_t*, esp_now_send_status_t){}

bool addPeer(const uint8_t mac[6]) {
  esp_now_peer_info_t p = {};
  memcpy(p.peer_addr, mac, 6);
  p.channel = 0; p.encrypt = false;
  if (esp_now_is_peer_exist(mac)) return true;
  return esp_now_add_peer(&p) == ESP_OK;
}

inline void sendTo(const uint8_t mac[6], const uint8_t* data, size_t len) {
  esp_now_send(mac, data, len);
}

inline void sendTL(const uint8_t mac[6], uint8_t target_id, TrafficCmd cmd) {
  tx.tl.type = MSG_TRAFFIC;
  tx.tl.cmd  = cmd;
  tx.tl.target_id = target_id;
  sendTo(mac, (uint8_t*)&tx, sizeof(msg_traffic_t));
}

// -------------------- Setup --------------------
void setup() {
  // Buttons
  pinMode(BTN_A_NEXT, INPUT_PULLUP);
  pinMode(BTN_B_NEXT, INPUT_PULLUP);
  pinMode(BTN_BOTH,   INPUT_PULLUP);
  btnA.last = btnA.stable = digitalRead(btnA.pin); btnA.tchg = millis();
  btnB.last = btnB.stable = digitalRead(btnB.pin); btnB.tchg = millis();
  btnAll.last = btnAll.stable = digitalRead(btnAll.pin); btnAll.tchg = millis();

  // Joystick
  pinMode(JOY_SW, INPUT_PULLUP);
  analogReadResolution(12);
  analogSetPinAttenuation(JOY_X, ADC_11db);
  analogSetPinAttenuation(JOY_Y, ADC_11db);

  // WiFi + ESP-NOW
  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) { while (1) delay(1000); }
  esp_now_register_send_cb(onNowSend);

  // Peers
  addPeer(CAR_MAC);
  addPeer(RAMZOR_A_MAC);
  addPeer(RAMZOR_B_MAC);
}

// -------------------- Helpers --------------------
inline bool handleButton(BtnState &b) {
  uint32_t now = millis();
  int r = digitalRead(b.pin);
  if (r != b.last) { b.last = r; b.tchg = now; }
  if ((now - b.tchg) >= DEBOUNCE_MS && b.stable != b.last) {
    int prev = b.stable;
    b.stable = b.last;
    if (prev == HIGH && b.stable == LOW) return true; // press
  }
  return false;
}

inline void joystickTask() {
  uint32_t now = millis();
  if (now - lastNowMs < NOW_PERIOD_MS) return;
  lastNowMs = now;

  tx.joy.type   = MSG_JOYSTICK;
  tx.joy.joy_x  = (int16_t)analogRead(JOY_X);
  tx.joy.joy_y  = (int16_t)analogRead(JOY_Y);
  tx.joy.joy_sw = (int8_t)digitalRead(JOY_SW);
  sendTo(CAR_MAC, (uint8_t*)&tx, sizeof(msg_joystick_t));
}

// -------------------- Loop --------------------
void loop() {
  joystickTask();

  // A only
  if (handleButton(btnA)) {
    sendTL(RAMZOR_A_MAC, TGT_A, TL_NEXT);
  }
  // B only
  if (handleButton(btnB)) {
    sendTL(RAMZOR_B_MAC, TGT_B, TL_NEXT);
  }
  // BOTH simultaneously
  if (handleButton(btnAll)) {
    sendTL(RAMZOR_A_MAC, TGT_ALL, TL_NEXT);
    sendTL(RAMZOR_B_MAC, TGT_ALL, TL_NEXT);
  }

  delay(1);
}
