# VisionChef GPU 서버 실행 가이드

## 사전 준비

- `.pem` 키 파일 보관 위치: `C:\VisionChef\JM\LLM\NVIDIA_A40_20260518_041725.pem`
- API 키: HuggingFace Token, YouTube Data API v3 키, ngrok 인증 토큰

---

## 1. SSH 접속

```powershell
ssh -i "C:\VisionChef\JM\LLM\NVIDIA_A40_20260518_041725.pem" -p 22 ubuntu@machine.runyour.ai
```

---

## 2. 최초 1회 설정 (처음 서버 빌릴 때만)

```bash
# 시스템 패키지 설치
sudo apt-get update && sudo apt-get install -y tmux portaudio19-dev python3-dev

# Python 패키지 설치
cd ~/JM
pip install -r requirements.txt
pip install accelerate mediapipe==0.10.14

# ngrok 설치
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update && sudo apt install ngrok -y

# ngrok 인증 (ngrok.com에서 토큰 발급)
ngrok config add-authtoken [ngrok_인증_토큰]
```

---

## 3. 서버 실행

```bash
# tmux 세션 시작
tmux new-session -s visionchef
```

### 창 0 — LLM 서버
```bash
cd ~/JM/LLM
export YOUTUBE_API_KEY="[유튜브_API_키]"
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

`Ctrl+B` → `C` (새 창)

### 창 1 — Frontend 서버
```bash
cd ~/JM/Web-main/Front
FRONT_HOST=0.0.0.0 python abc.py
```

`Ctrl+B` → `C` (새 창)

### 창 2 — ngrok
```bash
ngrok http 3000
```
→ 출력된 `https://xxxx.ngrok-free.app` 주소 복사

---

## 4. 접속 방법

| 기기 | 주소 | 용도 |
|------|------|------|
| 노트북 브라우저 | `https://[ngrok주소]/` | 질문 입력, 재료 확인 |
| 휴대폰 브라우저 | `https://[ngrok주소]/mobile` | 카메라 재료 인식 + YouTube 수신 |

> ngrok 주소는 서버 껐다 켤 때마다 바뀜

---

## 5. 노트북에서 SSH 터널로 접속하는 방법 (ngrok 대신)

별도 PowerShell 창에서 실행:

```powershell
ssh -i "C:\VisionChef\JM\LLM\NVIDIA_A40_20260518_041725.pem" -L 3000:127.0.0.1:3000 -p 22 ubuntu@machine.runyour.ai
```

그 후 노트북 브라우저에서:
```
http://localhost:3000
```

---

## 6. tmux 기본 단축키

| 단축키 | 동작 |
|--------|------|
| `Ctrl+B` → `C` | 새 창 열기 |
| `Ctrl+B` → `0~2` | 해당 번호 창으로 이동 |
| `Ctrl+B` → `D` | tmux 분리 (서버는 계속 실행) |
| `tmux attach -t visionchef` | tmux 다시 붙기 |

---

## 7. 서비스 흐름

```
휴대폰 카메라 → 재료 인식 → 노트북 화면에 실시간 표시
노트북에서 질문 → LLM 답변 + YouTube 영상 → 휴대폰 화면에 자동 재생
```
