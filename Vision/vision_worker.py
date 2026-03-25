import cv2
import time
import requests
import os # 파일 존재 확인을 위해 추가
from inference_sdk import InferenceHTTPClient

# ==========================================
# ⚙️ 설정
# ==========================================
LLM_SERVER_URL = "http://127.0.0.1:8000/vision"
ROBOFLOW_API_URL = "https://serverless.roboflow.com"
ROBOFLOW_API_KEY = "cHEA2qP7TzKjyoVBBUCP" 
ROBOFLOW_MODEL_ID = "food-ingredients-dataset/3"

# 🖼️ 카메라 대신 사용할 이미지 파일명
IMAGE_PATH = "test_img.jpg" 

def run_vision():
    # 이미지 파일이 실제로 있는지 먼저 확인
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ [Vision] '{IMAGE_PATH}' 파일을 찾을 수 없습니다. 사진을 Vision 폴더에 넣어주세요.")
        return

    print(f"👁️ [Vision] 이미지 모드 시작: {IMAGE_PATH}")
    client = InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=ROBOFLOW_API_KEY)
    
    last_ingredients = set()

    while True:
        # 1. 카메라 cap.read() 대신 이미지 파일을 읽어옴
        frame = cv2.imread(IMAGE_PATH)
        if frame is None:
            print("⚠️ [Vision] 이미지를 로드할 수 없습니다.")
            break

        frame = cv2.resize(frame, (640, 480))
        current_ingredients = []

        try:
            # 2. Roboflow API로 객체 탐지 요청
            results = client.infer(frame, model_id=ROBOFLOW_MODEL_ID)
            predictions = results.get("predictions", [])
            
            for p in predictions:
                if p['confidence'] > 0.4:
                    class_name = p['class']
                    current_ingredients.append(class_name)
                    
                    x, y, w, h = p['x'], p['y'], p['width'], p['height']
                    x1, y1 = int(x - w / 2), int(y - h / 2)
                    x2, y2 = int(x + w / 2), int(y + h / 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            unique_ingredients = list(set(current_ingredients))
            
            # 3. 서버 전송 (변화가 있을 때만)
            if set(unique_ingredients) != last_ingredients:
                print(f"📡 [Vision -> LLM] 업데이트 전송: {unique_ingredients}")
                try:
                    requests.post(LLM_SERVER_URL, json={"ingredients": unique_ingredients}, timeout=2)
                    last_ingredients = set(unique_ingredients)
                except requests.exceptions.RequestException:
                    print("⚠️ [Vision] LLM 서버 연결 불가")
            
        except Exception as e:
            print(f"⚠️ [Vision] 에러: {e}")

        # 화면에 사진 보여주기
        cv2.imshow("Robot Eye (Static Image Mode)", frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # 이미지 모드이므로 너무 자주 요청할 필요 없이 3초마다 확인
        time.sleep(3) 

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_vision()