import speech_recognition as sr
import os
import requests
import sys
import time
import webbrowser
from typing import Optional

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

# ==========================================
# ⚙️ 설정
# ==========================================
LLM_PORT = os.getenv("LLM_PORT", "8000")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", f"http://127.0.0.1:{LLM_PORT}").rstrip("/")
LLM_ASK_URL = f"{LLM_BASE_URL}/ask"
AUTO_OPEN_YOUTUBE = os.getenv("STT_AUTO_OPEN_YOUTUBE", "1").strip().lower() not in {"0", "false", "no", "off"}


def handle_video_recommendation(video: Optional[dict]):
    if not video:
        return

    title = video.get("title", "추천 영상")
    url = video.get("url")
    start_seconds = video.get("start_seconds", 0)
    match_source = video.get("match_source", "unknown")
    match_score = video.get("match_score", 0)

    print(f"🎬 [YouTube] {title}")
    print(f"⏱️ [YouTube] 시작 위치: {start_seconds}초, 기준: {match_source}, 점수: {match_score}")
    if url:
        print(f"🔗 [YouTube] {url}")
        if AUTO_OPEN_YOUTUBE:
            webbrowser.open(url)

def run_stt():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    # 🛠️ 여유 있는 인식을 위한 설정 값 조정
    # 사용자가 말을 멈춘 후 '아, 말이 끝났구나'라고 판단하기 전 대기 시간 (기본값 0.8초 -> 2.0초로 연장)
    recognizer.pause_threshold = 1.0
    # 에너지가 이 수치 이하면 침묵으로 간주 (주변 소음에 따라 자동 조절됨)
    recognizer.dynamic_energy_threshold = True

    print("👂 [STT] 음성 인식 모듈이 시작되었습니다. 마이크에 대고 말씀하세요!")
    print(f"🔗 [STT] LLM 서버: {LLM_ASK_URL}")
    
    with microphone as source:
        print("🔍 주변 소음에 적응 중... (잠시만 기다려주세요)")
        recognizer.adjust_for_ambient_noise(source, duration=1.5)
        
        while True:
            try:
                print("\n🎤 듣고 있습니다... (천천히 말씀하셔도 됩니다)")
                
                # timeout: 아무 소리도 안 들릴 때 대기하는 시간
                # phrase_time_limit: 한 번 말을 시작했을 때 최대 녹음 시간 (15초로 넉넉하게 설정)
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=15)
                
                print("🔍 [STT] 음성을 텍스트로 변환 중...")
                user_text = recognizer.recognize_google(audio, language='ko-KR')
                
                if user_text:
                    print(f"💬 [나]: {user_text}")
                    
                    try:
                        response = requests.post(LLM_ASK_URL, json={"user_text": user_text}, timeout=60)
                        if response.status_code == 200:
                            result = response.json()
                            print(f"🤖 [Chef]: {result.get('answer')}")
                            handle_video_recommendation(result.get("video_recommendation"))
                        else:
                            print(f"❌ [에러] 서버 응답 오류: {response.text}")
                    except requests.exceptions.RequestException:
                        print("⚠️ [에러] LLM 서버가 꺼져 있습니다.")

            except sr.WaitTimeoutError:
                continue 
            except sr.UnknownValueError:
                # 이 부분이 너무 자주 뜨면 무시하도록 처리 가능
                print("❓ [STT] 소리를 감지했지만 내용을 이해하지 못했습니다.")
            except Exception as e:
                print(f"⚠️ [STT] 오류 발생: {e}")

            time.sleep(1) # 루프 사이의 짧은 휴식

if __name__ == "__main__":
    run_stt()
