import cv2
import time
import requests
import os
from ultralytics import YOLO

# ==========================================
# ⚙️ 설정
# ==========================================
LLM_SERVER_URL = "http://127.0.0.1:8000/vision"

# 🧠 YOLO 모델 설정
YOLO_MODEL_PATH = "best.pt" 

# 🖼️ 카메라 대신 사용할 이미지 파일명
IMAGE_PATH = "test_img.jpg"

def run_vision():
    # 필수 파일 확인
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"❌ [Vision] '{YOLO_MODEL_PATH}' 파일을 찾을 수 없습니다. 훈련된 모델 파일을 Vision 폴더에 넣어주세요.")
        return
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ [Vision] '{IMAGE_PATH}' 파일을 찾을 수 없습니다. 이미지를 Vision 폴더에 넣어주세요.")
        return

    # YOLO 모델 로드
    try:
        model = YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        print(f"❌ [Vision] YOLO 모델 로드 중 오류 발생: {e}")
        print("💡 'ultralytics' 라이브러리가 설치되어 있는지 확인해주세요. (pip install ultralytics)")
        return
        
    print(f"👁️ [Vision] YOLO 모드 시작: {IMAGE_PATH}")
    
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print("⚠️ [Vision] 이미지를 로드할 수 없습니다.")
        return

    # 1. Ultralytics YOLO 모델로 객체 탐지
    try:
        results = model(frame)
    except Exception as e:
        print(f"⚠️ [Vision] 객체 탐지 중 오류 발생: {e}")
        return

    current_ingredients = []

    # 2. 결과 처리
    for box in results[0].boxes:
        confidence = box.conf[0]
        if confidence > 0.4:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            current_ingredients.append(class_name)

            # 바운딩 박스 좌표 및 시각화
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    unique_ingredients = list(set(current_ingredients))

    # 3. 서버 전송
    if unique_ingredients:
        print(f"📡 [Vision -> LLM] 업데이트 전송: {unique_ingredients}")
        try:
            requests.post(LLM_SERVER_URL, json={"ingredients": unique_ingredients}, timeout=3)
            print("✅ 전송 완료.")
        except requests.exceptions.RequestException:
            print("⚠️ [Vision] LLM 서버 연결 불가")
    else:
        print("ℹ️ [Vision] 탐지된 재료가 없습니다.")

    # 화면에 결과 보여주기
    cv2.imshow("Robot Eye (YOLO Mode)", frame)
    print("ℹ️ [Vision] 3초 후 창을 닫습니다...")
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_vision()
