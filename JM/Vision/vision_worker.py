import cv2
import time
import requests
import os
import sys
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import mediapipe as mp

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

# ==========================================
# ⚙️ 설정
# ==========================================
MODULE_DIR = Path(__file__).resolve().parent


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


LLM_PORT = os.getenv("LLM_PORT", "8000")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", f"http://127.0.0.1:{LLM_PORT}").rstrip("/")
LLM_SERVER_URL = f"{LLM_BASE_URL}/vision"

# 🧠 YOLO 모델 설정
YOLO_MODEL_PATH = Path(os.getenv("YOLO_MODEL_PATH", str(MODULE_DIR / "best.pt")))

# 📷 카메라 설정 (0: 기본 웹캠)
CAMERA_INDEX = _env_int("VISION_CAMERA_INDEX", 0)

# ⏱️ 안정화 대기 시간 (초)
STABLE_DURATION = _env_float("VISION_STABLE_DURATION", 3.0)
CAPTURE_DIR = Path(os.getenv("VISION_CAPTURE_DIR", str(MODULE_DIR / "captures")))

# ------------------------------------------
# ✋ MediaPipe 손 인식 설정
# ------------------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7, 
    max_num_hands=1  # 한 손만 인식
)
mp_draw = mp.solutions.drawing_utils

def detect_gesture(hand_landmarks):
    """
    손가락 관절의 Y 좌표를 기반으로 제스처를 판별합니다.
    """
    lm = hand_landmarks.landmark
    fingers = []
    
    # 검지(8), 중지(12), 약지(16), 소지(20)가 펴져 있는지 확인
    # Tip(끝)이 PIP(중간관절)와 MCP(뿌리관절)보다 위에 있으면 펴진 것으로 간주 (y좌표는 위가 0)
    for tip, pip, mcp in [(8,6,5), (12,10,9), (16,14,13), (20,18,17)]:
        if lm[tip].y < lm[pip].y and lm[tip].y < lm[mcp].y:
            fingers.append(1)
        else:
            fingers.append(0)
            
    # 엄지(4)가 펴져 있는지 확인 (엄지 끝이 검지 뿌리(5)보다 확연히 높이 있는지)
    thumb_up = lm[4].y < lm[3].y and lm[4].y < lm[5].y
    
    if fingers == [0, 0, 0, 0] and thumb_up:
        return "THUMBS_UP"  # 👍 엄지 척 (진행)
    elif fingers == [1, 1, 0, 0]:
        return "PEACE"      # ✌️ 브이 (캡처)
    elif fingers == [0, 0, 0, 0] and not thumb_up:
        return "FIST"       # ✊ 주먹 (거절/다시탐지)
    
    return "NONE"

def run_vision():
    if not YOLO_MODEL_PATH.exists():
        print(f"❌ [Vision] '{YOLO_MODEL_PATH}' 파일을 찾을 수 없습니다.")
        return

    # YOLO 모델 로드
    try:
        model = YOLO(str(YOLO_MODEL_PATH))
    except Exception as e:
        print(f"❌ [Vision] YOLO 모델 로드 오류: {e}")
        return
        
    print(f"👁️ [Vision] 실시간 웹캠 객체 및 제스처 탐지 시작")
    print(f"🔗 [Vision] LLM 서버: {LLM_SERVER_URL}")
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("⚠️ [Vision] 카메라를 열 수 없습니다.")
        return

    # 상태 관리
    STATE = "DETECTING"  # 현재 상태 (DETECTING: 재료 탐지 중 / WAITING_CONFIRM: 사용자 확인 대기)
    last_sent_ingredients = set()
    current_stable_ingredients = set()
    stable_start_time = time.time()
    
    capture_cooldown = 0
    feedback_message = ""
    feedback_time = 0

    print("========================================")
    print("👋 [사용 가능 제스처]")
    print(" 👍 엄지척 : 다음으로 진행 (LLM 확인 답변)")
    print(" ✊ 주먹   : 다시 탐지 (LLM 거절 답변)")
    print(" ✌️ 브이   : 화면 캡처")
    print("========================================")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time = time.time()
        current_gesture = "NONE"

        # ------------------------------------------
        # 1. MediaPipe 손 제스처 인식 로직
        # ------------------------------------------
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_hands = hands.process(rgb_frame)
        
        if result_hands.multi_hand_landmarks:
            for hand_landmarks in result_hands.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                current_gesture = detect_gesture(hand_landmarks)
                
                # 손모양을 화면에 표시
                cv2.putText(frame, f"Gesture: {current_gesture}", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        # ✌️ 브이 (화면 캡처 로직) - 상태에 상관없이 작동
        if current_gesture == "PEACE" and (current_time - capture_cooldown) > 3.0:
            CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
            filename = CAPTURE_DIR / f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(filename), frame)
            capture_cooldown = current_time
            feedback_message = f"Captured! ({filename})"
            feedback_time = current_time
            print(f"📸 [Vision] 화면이 캡처되었습니다: {filename}")

        # ------------------------------------------
        # 2. 상태별 처리 로직
        # ------------------------------------------
        if STATE == "DETECTING":
            # YOLO 객체 탐지
            results = model(frame, verbose=False)
            current_ingredients = []
            
            for box in results[0].boxes:
                confidence = float(box.conf[0])
                if confidence > 0.4:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    current_ingredients.append(class_name)

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            current_ingredients_set = set(current_ingredients)

            # 안정화 로직
            if current_ingredients_set != current_stable_ingredients:
                current_stable_ingredients = current_ingredients_set
                stable_start_time = current_time
            else:
                if len(current_stable_ingredients) > 0:
                    elapsed_stable_time = current_time - stable_start_time
                    if elapsed_stable_time >= STABLE_DURATION:
                        if current_stable_ingredients != last_sent_ingredients:
                            # 3초 안정화 완료 -> LLM에게 물어봐달라고 신호 보냄
                            print(f"📡 [Vision -> LLM] 확인 요청: {list(current_stable_ingredients)}")
                            try:
                                requests.post(LLM_SERVER_URL, json={
                                    "action": "ask_confirmation",
                                    "ingredients": list(current_stable_ingredients)
                                }, timeout=3)
                            except requests.exceptions.RequestException:
                                print("⚠️ [Vision] LLM 서버 연결 불가")
                            
                            STATE = "WAITING_CONFIRM"
                            feedback_message = "LLM Asking... (👍:Yes / ✊:No)"
                            feedback_time = current_time
                else:
                    if len(last_sent_ingredients) > 0 and (current_time - stable_start_time) >= STABLE_DURATION:
                        last_sent_ingredients = set()

            # 안정화 진행 시간 화면 표시
            if len(current_stable_ingredients) > 0 and current_stable_ingredients != last_sent_ingredients and STATE == "DETECTING":
                remain_time = max(0.0, STABLE_DURATION - (current_time - stable_start_time))
                cv2.putText(frame, f"Wait... {remain_time:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        elif STATE == "WAITING_CONFIRM":
            # 확인 대기 상태 화면 UI
            cv2.putText(frame, "[WAITING] LLM Asking...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            cv2.putText(frame, "[👍] YES / Proceed   [✊] NO / Retry", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            
            # 사용자 답변 제스처 처리
            if current_gesture == "THUMBS_UP":
                print("✅ [Vision] 👍 사용자 승인! 다음 단계 진행.")
                try:
                    requests.post(LLM_SERVER_URL, json={
                        "action": "confirm",
                        "ingredients": list(current_stable_ingredients)
                    }, timeout=3)
                except requests.exceptions.RequestException:
                    print("⚠️ [Vision] LLM 서버 연결 불가")
                last_sent_ingredients = current_stable_ingredients
                STATE = "DETECTING"
                current_stable_ingredients = set()
                feedback_message = "Confirmed!"
                feedback_time = current_time
                
            elif current_gesture == "FIST":
                print("🔄 [Vision] ✊ 사용자 거절! 탐지를 다시 시작합니다.")
                try:
                    requests.post(LLM_SERVER_URL, json={"action": "reject"}, timeout=3)
                except requests.exceptions.RequestException:
                    print("⚠️ [Vision] LLM 서버 연결 불가")
                STATE = "DETECTING"
                current_stable_ingredients = set()
                stable_start_time = current_time
                feedback_message = "Retrying..."
                feedback_time = current_time

        # 피드백 메시지를 2초간 띄워줌 (캡처완료, 확정됨 등)
        if current_time - feedback_time < 2.0:
            cv2.putText(frame, feedback_message, (10, frame.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Robot Eye (YOLO + Hands)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 [Vision] 종료됨.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_vision()
