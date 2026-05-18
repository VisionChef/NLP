import base64
from io import BytesIO
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if (PROJECT_ROOT / "NLP-jm" / "LLM").exists():
    NLP_ROOT = PROJECT_ROOT / "NLP-jm"
elif (PROJECT_ROOT / "LLM").exists():
    NLP_ROOT = PROJECT_ROOT
else:
    NLP_ROOT = PROJECT_ROOT / "NLP-jm"

LLM_ENV_PATH = NLP_ROOT / "LLM" / ".env"
GLOBAL_ENV_PATH = NLP_ROOT / ".env"
DEFAULT_MODEL_CANDIDATES = (
    NLP_ROOT / "Vision" / "best.pt",
    PROJECT_ROOT / "Vision" / "best.pt",
)
MODEL_PATH_ENV = os.getenv("YOLO_MODEL_PATH")
if MODEL_PATH_ENV:
    MODEL_PATH = Path(MODEL_PATH_ENV)
else:
    MODEL_PATH = next(
        (candidate for candidate in DEFAULT_MODEL_CANDIDATES if candidate.exists()),
        DEFAULT_MODEL_CANDIDATES[0],
    )

LLM_PORT = os.getenv("LLM_PORT", "8000")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", f"http://127.0.0.1:{LLM_PORT}").rstrip("/")
LLM_OFFLINE_MESSAGE = "LLM 서버를 자동으로 켜는 중입니다. 모델 로딩이 끝날 때까지 기다려주세요."
LLM_DIR = NLP_ROOT / "LLM"
AUTO_START_LLM = os.getenv("AUTO_START_LLM", "1").strip().lower() not in {"0", "false", "no", "off"}
LLM_START_TIMEOUT = int(os.getenv("LLM_START_TIMEOUT", "180"))
CONFIDENCE_THRESHOLD = float(os.getenv("VISION_CONFIDENCE", "0.4"))
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "gtts").strip().lower()
VARCO_TTS_URL = os.getenv("VARCO_TTS_URL", "").strip()
VARCO_API_KEY = os.getenv("VARCO_API_KEY", "").strip()

app = FastAPI(title="Cooking Agent Frontend")
_model = None
_llm_process = None
_llm_stdout = None
_llm_stderr = None


class IngredientsPayload(BaseModel):
    ingredients: list[str] = Field(default_factory=list)


class AskPayload(BaseModel):
    user_text: str
    ingredients: list[str] = Field(default_factory=list)


class TtsPayload(BaseModel):
    text: str


class SetupPayload(BaseModel):
    youtube_api_key: str | None = None
    hf_token: str | None = None
    setup_complete: bool = True


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        values[name.strip()] = value.strip().strip('"').strip("'")
    return values


def write_env_file(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{name}={value}" for name, value in values.items() if value is not None]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_dotenv_file(path: Path) -> None:
    for name, value in read_env_file(path).items():
        if name and name not in os.environ:
            os.environ[name] = value


def config_state() -> dict[str, Any]:
    llm_env = read_env_file(LLM_ENV_PATH)
    global_env = read_env_file(GLOBAL_ENV_PATH)
    youtube_key = os.getenv("YOUTUBE_API_KEY") or llm_env.get("YOUTUBE_API_KEY") or global_env.get("YOUTUBE_API_KEY")
    hf_token = os.getenv("HF_TOKEN") or llm_env.get("HF_TOKEN") or global_env.get("HF_TOKEN")
    setup_complete = (llm_env.get("SETUP_COMPLETE") or global_env.get("SETUP_COMPLETE")) == "1"
    return {
        "setup_complete": setup_complete,
        "youtube_api_key_configured": bool(youtube_key),
        "hf_token_configured": bool(hf_token),
        "llm_base_url": LLM_BASE_URL,
    }


load_dotenv_file(GLOBAL_ENV_PATH)
load_dotenv_file(LLM_ENV_PATH)


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=500, detail=f"YOLO model not found: {MODEL_PATH}")

        from ultralytics import YOLO

        _model = YOLO(str(MODEL_PATH))
    return _model


def decode_image(image_bytes: bytes) -> np.ndarray:
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Image could not be decoded.")
    return frame


def class_name(model: Any, class_id: int) -> str:
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    return str(names[class_id])


def detect_ingredients(frame: np.ndarray) -> dict[str, Any]:
    model = get_model()
    results = model(frame, verbose=False)
    detections = []

    for box in results[0].boxes:
        confidence = float(box.conf[0])
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        class_id = int(box.cls[0])
        name = class_name(model, class_id)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append(
            {
                "name": name,
                "confidence": round(confidence, 4),
                "box": [x1, y1, x2, y2],
            }
        )

        cv2.rectangle(frame, (x1, y1), (x2, y2), (242, 242, 242), 3)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), 1)
        cv2.putText(
            frame,
            f"{name} {confidence:.2f}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (20, 20, 20),
            4,
        )
        cv2.putText(
            frame,
            f"{name} {confidence:.2f}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (242, 242, 242),
            2,
        )

    ingredients = sorted({item["name"] for item in detections})
    ok, encoded = cv2.imencode(".jpg", frame)
    preview = base64.b64encode(encoded.tobytes()).decode("ascii") if ok else None
    return {"ingredients": ingredients, "detections": detections, "preview": preview}


def llm_error_detail(exc: requests.exceptions.RequestException) -> str:
    if isinstance(exc, requests.exceptions.ConnectionError):
        return LLM_OFFLINE_MESSAGE
    if isinstance(exc, requests.exceptions.Timeout):
        return "LLM 서버 응답이 늦습니다. 모델 로딩 중이면 조금 더 기다려주세요."
    return f"LLM 서버 요청 실패: {exc}"


def is_llm_ready(timeout: float = 1.0) -> bool:
    try:
        response = requests.get(f"{LLM_BASE_URL}/health", timeout=timeout)
        return response.ok
    except requests.exceptions.RequestException:
        return False


def is_llm_process_alive() -> bool:
    return _llm_process is not None and _llm_process.poll() is None


def stop_managed_llm() -> bool:
    global _llm_process
    if not is_llm_process_alive():
        return False

    _llm_process.terminate()
    try:
        _llm_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _llm_process.kill()
    _llm_process = None
    return True


def ensure_llm_started() -> bool:
    global _llm_process, _llm_stdout, _llm_stderr

    if is_llm_ready(timeout=0.5):
        return True
    if not AUTO_START_LLM or is_llm_process_alive() or not LLM_DIR.exists():
        return False

    log_dir = Path(os.getenv("COOKING_AGENT_LOG_DIR", r"C:\tmp"))
    log_dir.mkdir(parents=True, exist_ok=True)
    _llm_stdout = open(log_dir / "cooking_agent_llm.out.log", "a", encoding="utf-8", errors="replace")
    _llm_stderr = open(log_dir / "cooking_agent_llm.err.log", "a", encoding="utf-8", errors="replace")

    env = os.environ.copy()
    env.setdefault("LLM_HOST", "127.0.0.1")
    env["LLM_PORT"] = str(LLM_PORT)
    env.setdefault("LLM_BASE_URL", LLM_BASE_URL)

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    _llm_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(LLM_PORT)],
        cwd=str(LLM_DIR),
        env=env,
        stdout=_llm_stdout,
        stderr=_llm_stderr,
        creationflags=creationflags,
    )
    return False


def wait_for_llm(timeout: int = LLM_START_TIMEOUT) -> bool:
    ensure_llm_started()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_llm_ready(timeout=2):
            return True
        if _llm_process is not None and _llm_process.poll() is not None:
            return False
        time.sleep(2)
    return False


def post_llm(path: str, payload: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    if not wait_for_llm():
        raise HTTPException(status_code=503, detail=LLM_OFFLINE_MESSAGE)

    try:
        response = requests.post(f"{LLM_BASE_URL}{path}", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=503, detail=llm_error_detail(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="LLM 서버가 JSON이 아닌 응답을 반환했습니다.") from exc


def get_llm(path: str, timeout: int = 5) -> dict[str, Any]:
    ensure_llm_started()
    try:
        response = requests.get(f"{LLM_BASE_URL}{path}", timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=503, detail=llm_error_detail(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="LLM 서버가 JSON이 아닌 응답을 반환했습니다.") from exc


def get_llm_with_params(path: str, params: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    if not wait_for_llm():
        raise HTTPException(status_code=503, detail=LLM_OFFLINE_MESSAGE)

    try:
        response = requests.get(f"{LLM_BASE_URL}{path}", params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=503, detail=llm_error_detail(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="LLM 서버가 JSON이 아닌 응답을 반환했습니다.") from exc


def cancel_llm_output() -> dict[str, Any]:
    if not is_llm_ready(timeout=0.5):
        return {"status": "offline", "was_active": False}

    try:
        response = requests.post(f"{LLM_BASE_URL}/cancel", timeout=3)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=503, detail=llm_error_detail(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="LLM 서버가 JSON이 아닌 응답을 반환했습니다.") from exc


def clean_tts_text(text: str) -> str:
    clean = re.sub(r"[^\w\s가-힣?.!,]", "", text or "")
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:900]


def synthesize_gtts(text: str) -> tuple[bytes, str]:
    try:
        from gtts import gTTS
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="gTTS가 설치되어 있지 않습니다.") from exc

    fp = BytesIO()
    gTTS(text=text, lang="ko").write_to_fp(fp)
    return fp.getvalue(), "audio/mpeg"


def synthesize_varco_tts(text: str) -> tuple[bytes, str]:
    if not VARCO_TTS_URL or not VARCO_API_KEY:
        raise HTTPException(
            status_code=501,
            detail="VARCO_TTS_URL 또는 VARCO_API_KEY가 설정되지 않았습니다.",
        )

    try:
        response = requests.post(
            VARCO_TTS_URL,
            headers={
                "Authorization": f"Bearer {VARCO_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"text": text, "language": "ko-KR"},
            timeout=60,
        )
        response.raise_for_status()
        return response.content, response.headers.get("content-type", "audio/mpeg")
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"VARCO TTS 요청 실패: {exc}") from exc


def synthesize_tts_audio(text: str) -> tuple[bytes, str]:
    clean = clean_tts_text(text)
    if not clean:
        raise HTTPException(status_code=400, detail="TTS로 읽을 텍스트가 없습니다.")

    if TTS_PROVIDER == "varco":
        return synthesize_varco_tts(clean)
    if TTS_PROVIDER == "gtts":
        return synthesize_gtts(clean)

    raise HTTPException(status_code=501, detail=f"지원하지 않는 TTS_PROVIDER입니다: {TTS_PROVIDER}")


def shutdown_processes() -> None:
    time.sleep(0.7)
    if is_llm_ready(timeout=0.5):
        try:
            requests.post(f"{LLM_BASE_URL}/shutdown", timeout=2)
        except requests.exceptions.RequestException:
            pass
    stop_managed_llm()
    os._exit(0)


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(
        """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>흑백 셰프</title>
  <style>
    :root {
      color-scheme: dark;
      --black: #080808;
      --panel: #181818;
      --white: #f4f1e8;
      --ink: #171717;
      --paper: #f3efe5;
      --muted-dark: #a6a199;
      --muted-light: #625c53;
      --line-dark: rgba(255,255,255,.18);
      --line-light: rgba(0,0,0,.16);
      --gold: #d4b064;
      --red: #c94b42;
      --green: #7eaf74;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Arial, "Noto Sans KR", sans-serif;
      letter-spacing: 0;
      color: var(--white);
      background: #111;
    }
    button, input { font: inherit; letter-spacing: 0; }
    button {
      min-height: 40px;
      border: 1px solid var(--line-dark);
      border-radius: 8px;
      padding: 0 13px;
      color: var(--white);
      background: #151515;
      cursor: pointer;
      white-space: nowrap;
    }
    button.primary { color: #090909; border-color: var(--gold); background: var(--gold); font-weight: 700; }
    button.danger { border-color: rgba(201,75,66,.55); background: rgba(201,75,66,.2); }
    button.active { color: #090909; border-color: var(--white); background: var(--white); }
    button:disabled { opacity: .55; cursor: not-allowed; }
    input[type="text"], input[type="password"], input[type="file"] {
      width: 100%;
      min-height: 40px;
      border: 1px solid var(--line-light);
      border-radius: 8px;
      padding: 9px 10px;
      color: var(--ink);
      background: rgba(255,255,255,.86);
      outline: 0;
    }
    .screen {
      position: fixed;
      inset: 0;
      z-index: 10;
      display: grid;
      place-items: center;
      padding: 18px;
      background: #101010;
    }
    .screen[hidden], .app[hidden] { display: none; }
    .setup, .loader {
      width: min(520px, 100%);
      border: 1px solid var(--line-dark);
      border-radius: 8px;
      padding: 18px;
      background: #161616;
      display: grid;
      gap: 12px;
    }
    .setup h1, .loader h1 { margin: 0; font-size: 24px; }
    .setup p, .loader p { margin: 0; color: var(--muted-dark); line-height: 1.55; }
    .setup-row { display: grid; gap: 6px; }
    .setup-row label { font-size: 13px; color: var(--muted-dark); }
    .setup-actions { display: flex; gap: 8px; justify-content: flex-end; flex-wrap: wrap; }
    .loader-line {
      height: 8px;
      border-radius: 999px;
      background: #303030;
      overflow: hidden;
    }
    .loader-line::before {
      display: block;
      width: 42%;
      height: 100%;
      content: "";
      background: var(--gold);
      animation: load 1.2s infinite ease-in-out;
    }
    .loader.done .loader-line { display: none; }
    @keyframes load {
      0% { transform: translateX(-110%); }
      100% { transform: translateX(250%); }
    }
    .top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 20px;
      background: #0b0b0b;
      border-bottom: 1px solid var(--line-dark);
    }
    h1 { margin: 0; font-size: 24px; line-height: 1.1; }
    .states { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
    .states button { min-height: 28px; padding: 0 10px; font-size: 12px; }
    .pill {
      min-height: 28px;
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line-dark);
      border-radius: 999px;
      padding: 0 10px;
      color: var(--muted-dark);
      background: rgba(0,0,0,.32);
      font-size: 12px;
    }
    .pill.ok { color: #dff5d8; border-color: rgba(126,175,116,.72); }
    .pill.warn { color: #f3d999; border-color: rgba(212,176,100,.72); }
    .pill.error { color: #ffd2cd; border-color: rgba(201,75,66,.78); }
    main {
      width: min(1180px, 100%);
      margin: 0 auto;
      padding: 16px;
      display: grid;
      grid-template-columns: minmax(0, .95fr) minmax(380px, 1.05fr);
      gap: 14px;
      align-items: start;
      background: #181818;
      border-left: 1px solid var(--line-dark);
      border-right: 1px solid var(--line-dark);
      min-height: calc(100vh - 61px);
    }
    .card {
      border: 1px solid var(--line-dark);
      border-radius: 8px;
      background: #202020;
      overflow: hidden;
      box-shadow: none;
    }
    .card.light { color: var(--ink); border-color: var(--line-dark); background: #ebe7dc; }
    .bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 12px;
      border-bottom: 1px solid var(--line-dark);
    }
    .card.light .bar { border-color: var(--line-light); }
    .content { padding: 12px; display: grid; gap: 12px; }
    .tabs { display: inline-grid; grid-template-columns: 1fr 1fr; border: 1px solid var(--line-dark); border-radius: 8px; overflow: hidden; }
    .tabs button { min-width: 78px; border: 0; border-radius: 0; background: transparent; }
    .media {
      position: relative;
      min-height: 330px;
      display: grid;
      border: 1px solid var(--line-dark);
      border-radius: 8px;
      background: #080808;
      overflow: hidden;
    }
    video, .preview, .placeholder { grid-area: 1 / 1; width: 100%; height: 100%; min-height: 330px; object-fit: contain; }
    .placeholder {
      display: grid;
      place-items: center;
      color: transparent;
      background: #0b0b0b;
      -webkit-text-stroke: 1px rgba(212,176,100,.9);
      text-stroke: 1px rgba(212,176,100,.9);
      font-size: clamp(34px, 7vw, 72px);
      font-weight: 900;
    }
    .mode-panel[hidden], .preview[hidden], .placeholder[hidden] { display: none; }
    .actions, .voice-row, .manual-row { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
    .ask-row { display: grid; grid-template-columns: minmax(0, 1fr) auto auto; gap: 8px; }
    .voice-row { grid-template-columns: auto minmax(0, 1fr); }
    .manual-row { grid-template-columns: minmax(0, 1fr) auto auto; }
    .basic-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
    }
    .basic-grid label {
      min-height: 34px;
      display: flex;
      align-items: center;
      gap: 7px;
      border: 1px solid var(--line-light);
      border-radius: 8px;
      padding: 0 9px;
      background: rgba(255,255,255,.58);
      font-size: 13px;
    }
    .status { min-height: 22px; color: var(--muted-dark); font-size: 13px; line-height: 1.45; }
    .status.error { color: #ffb4ad; }
    .card.light .status { color: var(--muted-light); }
    .card.light .status.error { color: #a3261f; }
    .chips { min-height: 42px; display: flex; flex-wrap: wrap; gap: 8px; }
    .chip {
      min-height: 31px;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border: 1px solid var(--line-light);
      border-radius: 999px;
      padding: 0 10px;
      color: var(--ink);
      background: #fffaf0;
      font-size: 13px;
      font-weight: 700;
    }
    .chip button {
      min-height: 22px;
      width: 22px;
      padding: 0;
      color: var(--ink);
      border-color: rgba(0,0,0,.18);
      background: transparent;
    }
    .muted-box, .chat-log {
      border: 1px solid var(--line-light);
      border-radius: 8px;
      padding: 10px;
      color: var(--ink);
      background: rgba(255,255,255,.62);
      font-size: 14px;
      line-height: 1.6;
    }
    .chat-log {
      min-height: 230px;
      max-height: 420px;
      overflow: auto;
      display: grid;
      align-content: start;
      gap: 8px;
      white-space: pre-wrap;
    }
    .tracked-video {
      border: 1px solid var(--line-light);
      border-radius: 8px;
      padding: 10px;
      color: var(--ink);
      background: rgba(255,255,255,.76);
      display: grid;
      gap: 8px;
    }
    .tracked-video[hidden] { display: none; }
    .tracked-video-title {
      color: var(--muted-light);
      font-size: 13px;
      font-weight: 700;
    }
    .msg {
      max-width: 92%;
      border-radius: 8px;
      padding: 9px 10px;
      background: #fff;
      border: 1px solid rgba(0,0,0,.1);
    }
    .msg.user { justify-self: end; background: #171717; color: var(--white); }
    .msg.assistant { justify-self: start; background: #fffaf0; color: var(--ink); }
    .msg.system { justify-self: center; color: var(--muted-light); background: transparent; border: 0; text-align: center; }
    .video-card {
      margin-top: 8px;
      display: grid;
      gap: 7px;
    }
    .video-card iframe {
      width: min(420px, 100%);
      aspect-ratio: 16 / 9;
      border: 0;
      border-radius: 8px;
      background: #000;
    }
    .video-card a {
      color: #0b4a8b;
      font-size: 13px;
      word-break: break-all;
    }
    .video-answer { margin-top: 8px; }
    @media (max-width: 880px) {
      body { background: #0b0b0b; }
      .top { align-items: flex-start; flex-direction: column; }
      .states { justify-content: flex-start; }
      main { grid-template-columns: 1fr; padding: 12px; }
    }
    @media (max-width: 560px) {
      .bar, .actions, .ask-row, .voice-row, .manual-row { grid-template-columns: 1fr; }
      .basic-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .bar { align-items: stretch; flex-direction: column; }
      .media, video, .preview, .placeholder { min-height: 250px; }
      .actions button, .ask-row button, .voice-row button, .manual-row button { width: 100%; }
    }
  </style>
</head>
<body>
  <div id="setupScreen" class="screen" hidden>
    <div class="setup">
      <h1>초기 설정</h1>
      <p>시작 전에 API 키를 확인합니다. 이미 저장되어 있으면 바로 시작해도 됩니다.</p>
      <div class="setup-row">
        <label for="youtubeKey">YouTube API Key</label>
        <input id="youtubeKey" type="password" placeholder="선택 사항" />
      </div>
      <div class="setup-row">
        <label for="hfToken">Hugging Face Token</label>
        <input id="hfToken" type="password" placeholder="로컬 모델이 있으면 비워도 됨" />
      </div>
      <div id="setupStatus" class="status"></div>
      <div class="setup-actions">
        <button id="skipSetup" type="button">바로 시작</button>
        <button id="saveSetup" class="primary" type="button">저장하고 시작</button>
      </div>
    </div>
  </div>

  <div id="loadingScreen" class="screen" hidden>
    <div id="loadingBox" class="loader">
      <h1 id="loadingTitle">로딩 중</h1>
      <p id="loadingText">LLM 서버를 준비하고 있습니다.</p>
      <div class="loader-line"></div>
    </div>
  </div>

  <div id="appRoot" class="app" hidden>
    <header class="top">
      <h1>흑백 셰프</h1>
      <div class="states" aria-live="polite">
        <span id="visionState" class="pill warn">Vision 확인 중</span>
        <span id="llmState" class="pill warn">LLM loading</span>
        <span id="ragState" class="pill warn">RAG loading</span>
        <span id="youtubeState" class="pill warn">YouTube 확인 중</span>
        <button id="settingsButton" type="button">설정</button>
        <button id="shutdownButton" class="danger" type="button">종료</button>
      </div>
    </header>

    <main>
      <section class="card">
        <div class="bar">
          <strong>재료 인식</strong>
          <div class="tabs" role="tablist" aria-label="입력 방식">
            <button id="cameraTab" class="active" type="button" data-mode="camera">카메라</button>
            <button id="uploadTab" type="button" data-mode="upload">업로드</button>
          </div>
        </div>

        <div class="content">
          <div id="uploadPanel" class="mode-panel" hidden>
            <input id="imageFile" type="file" accept="image/*" />
          </div>
          <div class="media">
            <video id="camera" autoplay playsinline muted></video>
            <img id="previewImage" class="preview" hidden alt="인식 결과" />
            <div id="placeholder" class="placeholder">B/W</div>
          </div>
          <div class="actions">
            <button id="startCamera" type="button">카메라 켜기</button>
            <button id="detectButton" class="primary" type="button">인식</button>
          </div>
          <div id="status" class="status" aria-live="polite"></div>
        </div>
      </section>

      <section class="card light">
        <div class="bar">
          <strong>재료와 대화</strong>
          <div class="actions">
            <button id="confirmButton" class="primary" type="button">재료 확정</button>
            <button id="clearButton" class="danger" type="button">초기화</button>
          </div>
        </div>

        <div class="content">
          <div class="manual-row">
            <input id="manualIngredient" type="text" placeholder="재료 직접 추가" />
            <button id="addIngredientButton" type="button">추가</button>
            <button id="addSelectedBasicsButton" type="button">선택 추가</button>
          </div>
          <div id="basicIngredients" class="basic-grid"></div>
          <div id="ingredients" class="chips"></div>
          <div id="detections" class="status"></div>

          <div class="voice-row">
            <button id="voiceButton" type="button">말하기</button>
            <div id="voiceText" class="muted-box">버튼을 누르면 한 번만 듣습니다.</div>
          </div>

          <div class="ask-row">
            <input id="askText" type="text" placeholder="질문 입력" />
            <button id="askButton" type="button">전송</button>
            <button id="stopButton" class="danger" type="button" disabled>말 끊기</button>
          </div>
          <div id="trackedVideo" class="tracked-video" hidden></div>
          <div id="chatLog" class="chat-log"></div>
        </div>
      </section>
    </main>
  </div>

  <canvas id="captureCanvas" hidden></canvas>
  <script>
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const DEFAULT_INGREDIENTS = ["소금", "설탕", "간장", "식용유", "들기름", "참기름", "고춧가루", "밥", "마늘", "파", "후춧가루"];
    const state = {
      mode: "camera",
      stream: null,
      ingredients: [],
      messages: [],
      recognition: null,
      pendingVoiceText: "",
      listening: false,
      processingVoice: false,
      asking: false,
      speaking: false,
      speechSeq: 0,
      speechResolve: null,
      audioPlayer: null,
      audioUrl: "",
      askAbortController: null,
      videoTrackTimer: null,
      videoTrackSeq: 0,
      trackedVideo: null,
      trackedVideoText: ""
    };
    const els = {};

    function bindElements() {
      [
        "setupScreen", "loadingScreen", "appRoot", "loadingText", "youtubeKey", "hfToken",
        "loadingBox", "loadingTitle",
        "setupStatus", "visionState", "llmState", "ragState", "youtubeState", "camera", "uploadPanel",
        "imageFile", "previewImage", "placeholder", "status", "ingredients", "detections",
        "chatLog", "askText", "askButton", "stopButton", "trackedVideo", "manualIngredient", "voiceButton", "voiceText", "captureCanvas",
        "settingsButton", "shutdownButton", "basicIngredients"
      ].forEach((id) => { els[id] = document.getElementById(id); });
      els.canvas = els.captureCanvas;
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, (char) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      })[char]);
    }

    async function fetchJson(url, options = {}) {
      const response = await fetch(url, options);
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || "요청 실패");
      return data;
    }

    function setStatus(text, error = false) {
      els.status.textContent = text;
      els.status.classList.toggle("error", error);
    }

    function setPill(element, text, tone = "") {
      element.textContent = text;
      element.className = `pill ${tone}`.trim();
    }

    function showSetup() {
      els.setupScreen.hidden = false;
      els.loadingScreen.hidden = true;
      els.appRoot.hidden = true;
    }

    function showLoading(text) {
      els.setupScreen.hidden = true;
      els.loadingScreen.hidden = false;
      els.appRoot.hidden = true;
      els.loadingBox.classList.remove("done");
      els.loadingTitle.textContent = "로딩 중";
      els.loadingText.textContent = text;
    }

    function showFinished(text) {
      els.setupScreen.hidden = true;
      els.loadingScreen.hidden = false;
      els.appRoot.hidden = true;
      els.loadingBox.classList.add("done");
      els.loadingTitle.textContent = "종료 완료";
      els.loadingText.textContent = text;
    }

    function showApp() {
      els.setupScreen.hidden = true;
      els.loadingScreen.hidden = true;
      els.appRoot.hidden = false;
    }

    async function saveSetup(skip = false) {
      els.setupStatus.textContent = "저장 중...";
      try {
        await fetchJson("/api/config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            youtube_api_key: skip ? null : els.youtubeKey.value.trim(),
            hf_token: skip ? null : els.hfToken.value.trim(),
            setup_complete: true
          })
        });
        await fetchJson("/api/restart-llm", { method: "POST" });
        await bootApp();
      } catch (error) {
        els.setupStatus.textContent = error.message;
        els.setupStatus.classList.add("error");
      }
    }

    async function bootApp() {
      showLoading("LLM 서버를 켜고 있습니다. 첫 실행은 모델 로딩 때문에 시간이 걸립니다.");
      for (let i = 0; i < 120; i += 1) {
        const llm = await fetchJson("/api/llm-health");
        if (llm.llm_loaded) {
          showApp();
          renderBasicIngredients();
          renderIngredients();
          renderChat();
          await refreshHealth();
          return;
        }
        els.loadingText.textContent = llm.llm_process_alive
          ? "LLM 모델 로딩 중입니다."
          : "LLM 서버를 시작하는 중입니다.";
        await new Promise((resolve) => setTimeout(resolve, 2500));
      }
      els.loadingText.textContent = "LLM 로딩이 오래 걸립니다. 잠시 후 새로고침하거나 로그를 확인하세요.";
    }

    async function init() {
      bindElements();
      document.getElementById("saveSetup").addEventListener("click", () => saveSetup(false));
      document.getElementById("skipSetup").addEventListener("click", () => saveSetup(true));
      els.settingsButton.addEventListener("click", showSetup);
      els.shutdownButton.addEventListener("click", shutdownApp);

      const config = await fetchJson("/api/config");
      els.setupStatus.textContent = config.youtube_api_key_configured
        ? "YouTube API 키가 저장되어 있습니다. 바로 시작해도 됩니다."
        : "YouTube 영상 추천을 쓰려면 API 키를 입력하세요.";
      els.youtubeKey.placeholder = config.youtube_api_key_configured ? "이미 저장됨" : "선택 사항";
      els.hfToken.placeholder = config.hf_token_configured ? "이미 저장됨" : "로컬 모델이 있으면 비워도 됨";
      showSetup();
      wireMainEvents();
    }

    function wireMainEvents() {
      document.querySelectorAll("[data-mode]").forEach((button) => {
        button.addEventListener("click", () => setMode(button.dataset.mode));
      });
      document.getElementById("startCamera").addEventListener("click", startCamera);
      document.getElementById("detectButton").addEventListener("click", async () => {
        try {
          if (state.mode === "camera") await detectFromCamera();
          else await detectFromUpload();
        } catch (error) {
          setStatus(error.message, true);
        }
      });
      document.getElementById("confirmButton").addEventListener("click", async () => {
        try { await confirmIngredients({ speak: true }); } catch (error) { setStatus(error.message, true); }
      });
      document.getElementById("clearButton").addEventListener("click", resetSession);
      els.askButton.addEventListener("click", async () => {
        try { await askChef(undefined, { speak: true }); } catch (error) { setStatus(error.message, true); addMessage("system", error.message); }
      });
      els.stopButton.addEventListener("click", stopConversation);
      document.getElementById("addIngredientButton").addEventListener("click", addManualIngredient);
      document.getElementById("addSelectedBasicsButton").addEventListener("click", addSelectedBasicIngredients);
      els.manualIngredient.addEventListener("keydown", (event) => {
        if (event.key === "Enter") addManualIngredient();
      });
      els.askText.addEventListener("keydown", async (event) => {
        if (event.key !== "Enter") return;
        event.preventDefault();
        try { await askChef(undefined, { speak: true }); } catch (error) { setStatus(error.message, true); addMessage("system", error.message); }
      });
      els.voiceButton.addEventListener("click", startOneShotVoice);
      if (!SpeechRecognition) {
        els.voiceButton.disabled = true;
        els.voiceText.textContent = "Chrome 또는 Edge 필요";
      }
      window.setInterval(refreshHealth, 10000);
    }

    async function refreshHealth() {
      try {
        const front = await fetchJson("/health");
        setPill(els.visionState, front.model_found ? "Vision ready" : "Vision model missing", front.model_found ? "ok" : "error");
      } catch {
        setPill(els.visionState, "Vision error", "error");
      }
      try {
        const llm = await fetchJson("/api/llm-health");
        const llmLabel = llm.llm_generating ? "LLM answering" : (llm.llm_loaded ? "LLM ready" : "LLM loading");
        setPill(els.llmState, llmLabel, llm.llm_loaded ? "ok" : "warn");
        const ragLabel = llm.rag_mode === "fallback" ? "RAG fallback" : "RAG ready";
        setPill(els.ragState, llm.rag_loaded ? ragLabel : "RAG loading", llm.rag_loaded ? "ok" : "warn");
        const youtubeLabel = llm.youtube_enabled
          ? (llm.llm_loaded ? "YouTube ready" : "YouTube key 저장됨")
          : "YouTube key 없음";
        setPill(els.youtubeState, youtubeLabel, llm.youtube_enabled ? "ok" : "warn");
      } catch {
        setPill(els.llmState, "LLM loading", "warn");
        setPill(els.ragState, "RAG loading", "warn");
        setPill(els.youtubeState, "YouTube 확인 중", "warn");
      }
    }

    function updateStopButton() {
      els.stopButton.disabled = !state.asking && !state.speaking;
    }

    function setAsking(active) {
      state.asking = active;
      els.askButton.disabled = active;
      updateStopButton();
    }

    function setSpeaking(active) {
      state.speaking = active;
      updateStopButton();
    }

    function stopAudioPlayback() {
      state.speechSeq += 1;
      if (state.audioPlayer) {
        state.audioPlayer.pause();
        state.audioPlayer.removeAttribute("src");
        state.audioPlayer.load();
        state.audioPlayer = null;
      }
      if (state.audioUrl) {
        URL.revokeObjectURL(state.audioUrl);
        state.audioUrl = "";
      }
      if (state.speechResolve) {
        state.speechResolve();
        state.speechResolve = null;
      }
      setSpeaking(false);
    }

    function showPreview(src) {
      if (src) {
        els.previewImage.src = src;
        els.previewImage.hidden = false;
        els.placeholder.hidden = true;
      } else {
        els.previewImage.hidden = true;
        els.previewImage.removeAttribute("src");
        els.placeholder.hidden = false;
      }
    }

    function setMode(mode) {
      state.mode = mode;
      document.querySelectorAll("[data-mode]").forEach((button) => {
        button.classList.toggle("active", button.dataset.mode === mode);
      });
      els.uploadPanel.hidden = mode !== "upload";
      setStatus("");
    }

    async function startCamera() {
      if (state.stream) return;
      try {
        state.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
        els.camera.srcObject = state.stream;
        els.placeholder.hidden = true;
        setStatus("카메라 준비됨");
      } catch (error) {
        setStatus(`카메라를 열 수 없습니다: ${error.message}`, true);
      }
    }

    function setIngredients(items) {
      const seen = new Set();
      state.ingredients = items
        .map((item) => String(item).trim())
        .filter((item) => item && !seen.has(item) && seen.add(item));
      renderIngredients();
    }

    function addManualIngredient() {
      const value = els.manualIngredient.value.trim();
      if (!value) return;
      setIngredients([...state.ingredients, value]);
      els.manualIngredient.value = "";
    }

    function selectedBasicIngredients() {
      return [...els.basicIngredients.querySelectorAll("input:checked")]
        .map((input) => input.value);
    }

    function ingredientsForRequest() {
      const seen = new Set();
      return [...state.ingredients, ...selectedBasicIngredients()]
        .map((item) => String(item).trim())
        .filter((item) => item && !seen.has(item) && seen.add(item));
    }

    function renderBasicIngredients() {
      els.basicIngredients.innerHTML = DEFAULT_INGREDIENTS
        .map((item) => `
          <label>
            <input type="checkbox" value="${escapeHtml(item)}" />
            ${escapeHtml(item)}
          </label>
        `)
        .join("");
    }

    function addSelectedBasicIngredients() {
      const selected = selectedBasicIngredients();
      if (!selected.length) {
        setStatus("추가할 기본 재료를 체크하세요.", true);
        return;
      }
      setIngredients([...state.ingredients, ...selected]);
      els.basicIngredients.querySelectorAll("input:checked").forEach((input) => {
        input.checked = false;
      });
      setStatus(`${selected.length}개 기본 재료 추가됨`);
    }

    function removeIngredient(item) {
      setIngredients(state.ingredients.filter((ingredient) => ingredient !== item));
    }

    function renderIngredients() {
      els.ingredients.innerHTML = state.ingredients.length
        ? state.ingredients.map((item) => `<span class="chip">${escapeHtml(item)}<button type="button" data-remove="${escapeHtml(item)}">x</button></span>`).join("")
        : `<span class="status">재료 없음</span>`;
      els.ingredients.querySelectorAll("[data-remove]").forEach((button) => {
        button.addEventListener("click", () => removeIngredient(button.dataset.remove));
      });
    }

    function renderDetection(data) {
      setIngredients(data.ingredients || []);
      els.detections.innerHTML = (data.detections || [])
        .map((item) => `${escapeHtml(item.name)} · ${(item.confidence * 100).toFixed(1)}%`)
        .join("<br>");
      showPreview(data.preview ? `data:image/jpeg;base64,${data.preview}` : "");
    }

    async function detectBlob(blob, filename) {
      if (!blob) throw new Error("이미지를 캡처하지 못했습니다.");
      const formData = new FormData();
      formData.append("file", blob, filename);
      setStatus("인식 중...");
      const data = await fetchJson("/api/detect", { method: "POST", body: formData });
      renderDetection(data);
      setStatus(`${state.ingredients.length}개 재료 인식됨`);
    }

    async function detectFromCamera() {
      if (!state.stream) await startCamera();
      if (!state.stream) return;
      const canvas = els.canvas;
      canvas.width = els.camera.videoWidth || 1280;
      canvas.height = els.camera.videoHeight || 720;
      canvas.getContext("2d").drawImage(els.camera, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
      await detectBlob(blob, "camera.jpg");
    }

    async function detectFromUpload() {
      const file = els.imageFile.files[0];
      if (!file) {
        setStatus("파일을 선택하세요.", true);
        return;
      }
      await detectBlob(file, file.name || "upload.jpg");
    }

    function withQueryParam(url, name, value) {
      if (!url) return "";
      const separator = url.includes("?") ? "&" : "?";
      return `${url}${separator}${encodeURIComponent(name)}=${encodeURIComponent(value)}`;
    }

    function renderVideo(video, autoplay = false) {
      if (!video?.url) return "";
      let embedUrl = (video.embed_url || "")
        .replace(/([?&])autoplay=1&?/, "$1")
        .replace(/[?&]$/, "");
      if (autoplay && embedUrl) embedUrl = withQueryParam(embedUrl, "autoplay", "1");
      const title = video.title || "YouTube 영상";
      const iframe = embedUrl
        ? `<iframe src="${escapeHtml(embedUrl)}" title="${escapeHtml(title)}" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>`
        : "";
      return `
        <div class="video-card">
          ${iframe}
          <a href="${escapeHtml(video.url)}" target="_blank" rel="noreferrer">${escapeHtml(title)}</a>
        </div>
      `;
    }

    function looksLikeVideoQuery(text) {
      const lower = text.trim().toLowerCase();
      const compact = lower.replace(/\\s+/g, "");
      return /(유튜브|영상|동영상|보여줘|보여주세요|틀어줘|찾아줘|추천해줘|시연|demo|recipe)/.test(lower)
        || /(자르는법|써는법|만드는법|하는법|손질|썰기|자르|다지|볶는법|삶는법|끓이는법|불리는법|불리는|불리|굽는법|튀기는법|까는법|씻는법|익히는법)/.test(compact);
    }

    function clearTrackedVideo() {
      if (state.videoTrackTimer) window.clearTimeout(state.videoTrackTimer);
      state.videoTrackTimer = null;
      state.trackedVideo = null;
      state.trackedVideoText = "";
      state.videoTrackSeq += 1;
      els.trackedVideo.hidden = true;
      els.trackedVideo.innerHTML = "";
    }

    function scheduleVideoTracking() {
      const text = els.askText.value.trim();
      if (state.videoTrackTimer) window.clearTimeout(state.videoTrackTimer);
      if (text.length < 4 || !looksLikeVideoQuery(text)) {
        clearTrackedVideo();
        return;
      }

      const seq = state.videoTrackSeq + 1;
      state.videoTrackSeq = seq;
      state.videoTrackTimer = window.setTimeout(() => trackVideoForInput(text, seq), 700);
    }

    async function trackVideoForInput(text, seq) {
      els.trackedVideo.hidden = false;
      els.trackedVideo.innerHTML = `<div class="tracked-video-title">영상 찾는 중</div>`;

      try {
        const data = await fetchJson(`/api/youtube-preview?query=${encodeURIComponent(text)}`);
        if (seq !== state.videoTrackSeq || els.askText.value.trim() !== text) return;

        const video = data.video_recommendation || null;
        state.trackedVideo = video;
        state.trackedVideoText = video ? text : "";

        if (video?.url) {
          els.trackedVideo.innerHTML = `<div class="tracked-video-title">입력 내용과 맞는 영상</div>${renderVideo(video)}`;
        } else if (data.youtube_status?.requested && data.youtube_status?.message) {
          els.trackedVideo.innerHTML = `<div class="tracked-video-title">${escapeHtml(data.youtube_status.message)}</div>`;
        } else {
          clearTrackedVideo();
        }
      } catch (error) {
        if (seq !== state.videoTrackSeq) return;
        state.trackedVideo = null;
        state.trackedVideoText = "";
        els.trackedVideo.innerHTML = `<div class="tracked-video-title">${escapeHtml(error.message)}</div>`;
      }
    }

    function addMessage(role, text, video = null, options = {}) {
      if (!text) return;
      state.messages.push({ role, text, video, videoAutoplay: Boolean(options.videoAutoplay) });
      renderChat();
    }

    function renderMessage(msg) {
      const text = escapeHtml(msg.text);
      const video = renderVideo(msg.video, msg.videoAutoplay);
      if (msg.role === "assistant" && msg.video?.url) {
        return `<div class="msg ${msg.role}">${video}<div class="video-answer">${text}</div></div>`;
      }
      return `<div class="msg ${msg.role}">${text}${video}</div>`;
    }

    function renderChat() {
      els.chatLog.innerHTML = state.messages.length
        ? state.messages.map(renderMessage).join("")
        : `<div class="msg system">대화가 여기에 표시됩니다.</div>`;
      state.messages.forEach((msg) => { msg.videoAutoplay = false; });
      els.chatLog.scrollTop = els.chatLog.scrollHeight;
    }

    function formatRagMatches(matches = []) {
      if (!matches.length) return "";
      const lines = matches.slice(0, 3).map((match) => {
        const raw = Number(match.similarity || 0);
        const percent = Math.max(0, Math.min(100, raw <= 1 ? raw * 100 : raw));
        return `${match.title || "레시피"} ${percent.toFixed(1)}%`;
      });
      return `RAG 매칭: ${lines.join(", ")}`;
    }

    async function confirmIngredients(options = {}) {
      const ingredients = ingredientsForRequest();
      if (!ingredients.length) throw new Error("확정할 재료가 없습니다.");
      setIngredients(ingredients);
      setStatus("LLM 서버로 전송 중...");
      const data = await fetchJson("/api/confirm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ingredients })
      });
      const message = data.message || `확정됨 · 레시피 ${data.recipes_found ?? 0}개`;
      setStatus(message);
      addMessage("assistant", message);
      const ragSummary = formatRagMatches(data.rag_matches || []);
      if (ragSummary) addMessage("system", ragSummary);
      if (options.speak && message) await speakText(message);
      await refreshHealth();
      return data;
    }

    async function rejectIngredients() {
      const data = await fetchJson("/api/reject", { method: "POST" });
      clearAll(false);
      const message = data.message || "재료를 다시 인식합니다.";
      setStatus(message);
      addMessage("assistant", message);
      return data;
    }

    async function resetSession() {
      try {
        await rejectIngredients();
      } catch (error) {
        clearAll();
        setStatus(`화면만 초기화됨: ${error.message}`, true);
      }
    }

    async function askChef(text = els.askText.value.trim(), options = {}) {
      if (!text || state.asking) return null;
      const trackedVideo = state.trackedVideoText === text ? state.trackedVideo : null;
      const ingredients = ingredientsForRequest();
      els.askText.value = "";
      clearTrackedVideo();
      addMessage("user", text);
      setAsking(true);
      setStatus("답변 생성 중...");
      state.askAbortController = new AbortController();
      try {
        const data = await fetchJson("/api/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ user_text: text, ingredients }),
          signal: state.askAbortController.signal
        });
        if (data.cancelled) {
          addMessage("system", data.answer || "응답 생성을 중단했습니다.");
          setStatus("응답 중단됨");
          return data;
        }

        let answer = data.answer || "";
        const video = data.video_recommendation || trackedVideo || null;
        addMessage("assistant", answer, video, { videoAutoplay: Boolean(video?.url) });
        if (data.youtube_status?.requested && data.youtube_status?.message && !video?.url) {
          addMessage("system", data.youtube_status.message);
        }
        if (data.youtube_status?.requested && !video?.url) {
          addMessage("system", "영상이 보이지 않으면 설정의 YouTube API 키와 API 할당량을 확인하세요.");
        }
        if (options.speak && !video?.url && answer) {
          await speakText(answer);
        }
        setStatus("");
        await refreshHealth();
        return data;
      } catch (error) {
        if (error.name === "AbortError") {
          addMessage("system", "응답 생성을 중단했습니다.");
          setStatus("응답 중단됨");
          return { cancelled: true, answer: "" };
        }
        throw error;
      } finally {
        state.askAbortController = null;
        setAsking(false);
      }
    }

    async function stopConversation() {
      if (!state.asking && !state.speaking) return;
      const wasAsking = state.asking;
      els.stopButton.disabled = true;
      setStatus(wasAsking ? "답변 생성을 중단하는 중..." : "음성을 중단했습니다.");
      stopAudioPlayback();
      if (wasAsking && state.askAbortController) state.askAbortController.abort();
      try {
        await fetchJson("/api/cancel", { method: "POST" });
      } catch (error) {
        if (wasAsking) setStatus(error.message, true);
        updateStopButton();
      }
    }

    async function shutdownApp() {
      if (!window.confirm("앱을 종료할까요?")) return;
      stopAudioPlayback();
      if (state.stream) {
        state.stream.getTracks().forEach((track) => track.stop());
        state.stream = null;
      }
      showLoading("앱을 종료하는 중입니다. 서버와 LLM을 차례대로 정리하고 있습니다.");
      const slowShutdown = window.setTimeout(() => {
        els.loadingText.textContent = "종료가 조금 오래 걸립니다. LLM 프로세스를 정리하는 중입니다.";
      }, 2500);
      try {
        await fetchJson("/api/shutdown", { method: "POST" });
        window.clearTimeout(slowShutdown);
        showFinished("앱이 종료되었습니다. 이제 이 창을 닫아도 됩니다.");
      } catch (error) {
        window.clearTimeout(slowShutdown);
        showFinished("종료 신호를 보냈습니다. 창을 닫아도 됩니다.");
      }
    }

    function clearAll(clearStatus = true) {
      state.ingredients = [];
      state.messages = [];
      els.detections.innerHTML = "";
      els.voiceText.textContent = "버튼을 누르면 한 번만 듣습니다.";
      clearTrackedVideo();
      showPreview("");
      renderIngredients();
      renderChat();
      if (clearStatus) setStatus("");
    }

    function setupRecognition() {
      if (!SpeechRecognition || state.recognition) return;
      const recognition = new SpeechRecognition();
      recognition.lang = "ko-KR";
      recognition.interimResults = true;
      recognition.continuous = false;
      recognition.onstart = () => {
        state.listening = true;
        els.voiceButton.classList.add("active");
        els.voiceButton.textContent = "듣는 중";
        els.voiceText.textContent = "말씀하세요.";
      };
      recognition.onresult = (event) => {
        let finalText = "";
        let interimText = "";
        for (let i = event.resultIndex; i < event.results.length; i += 1) {
          const text = event.results[i][0].transcript.trim();
          if (event.results[i].isFinal) finalText += `${text} `;
          else interimText += text;
        }
        if (interimText) els.voiceText.textContent = interimText;
        if (finalText.trim()) state.pendingVoiceText = finalText.trim();
      };
      recognition.onerror = (event) => {
        els.voiceText.textContent = event.error === "no-speech" ? "수음된 말이 없어 요청하지 않았습니다." : `STT 오류: ${event.error}`;
      };
      recognition.onend = () => {
        state.listening = false;
        els.voiceButton.classList.remove("active");
        els.voiceButton.textContent = "말하기";
        const text = (state.pendingVoiceText || "").trim();
        state.pendingVoiceText = "";
        if (text) handleVoiceCommand(text);
      };
      state.recognition = recognition;
    }

    function startOneShotVoice() {
      if (!SpeechRecognition) {
        els.voiceText.textContent = "Chrome 또는 Edge에서 음성 인식을 사용할 수 있습니다.";
        return;
      }
      if (state.processingVoice) return;
      setupRecognition();
      if (state.listening) {
        state.recognition.stop();
        return;
      }
      try {
        state.recognition.start();
      } catch (error) {
        if (!String(error.message || "").includes("already started")) els.voiceText.textContent = error.message;
      }
    }

    function isConfirmText(text) {
      return /(맞아|맞습니다|네|응|오케이|좋아|확정|확인|진행)/.test(text.replace(/\\s/g, ""));
    }

    function isRejectText(text) {
      return /(아니|아니야|틀려|다시|취소|초기화|재인식)/.test(text.replace(/\\s/g, ""));
    }

    async function speakText(text) {
      if (!text) return;
      stopAudioPlayback();
      const speechSeq = state.speechSeq + 1;
      state.speechSeq = speechSeq;
      setSpeaking(true);
      try {
        const response = await fetch("/api/tts", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text })
        });
        const errorData = response.ok ? null : await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(errorData?.detail || "TTS 생성 실패");
        if (state.speechSeq !== speechSeq) return;

        const blob = await response.blob();
        if (state.speechSeq !== speechSeq) return;

        const audioUrl = URL.createObjectURL(blob);
        const audio = new Audio(audioUrl);
        state.audioUrl = audioUrl;
        state.audioPlayer = audio;
        await new Promise((resolve, reject) => {
          const finish = () => {
            if (state.speechResolve === finish) state.speechResolve = null;
            resolve();
          };
          state.speechResolve = finish;
          audio.onended = finish;
          audio.onerror = () => reject(new Error("TTS 오디오 재생 실패"));
          audio.play().catch(reject);
        });
      } catch (error) {
        if (state.speechSeq === speechSeq) setStatus(`TTS 오류: ${error.message}`, true);
      } finally {
        if (state.speechSeq === speechSeq) {
          if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
          state.audioUrl = "";
          state.audioPlayer = null;
          state.speechResolve = null;
          setSpeaking(false);
        }
      }
    }

    async function handleVoiceCommand(text) {
      state.processingVoice = true;
      els.voiceText.textContent = text;
      try {
        let data = null;
        if (state.ingredients.length && isConfirmText(text)) {
          data = await confirmIngredients();
          await speakText(data.message || "재료를 확정했습니다.");
        } else if (state.ingredients.length && isRejectText(text)) {
          data = await rejectIngredients();
          await speakText(data.message || "다시 인식하겠습니다.");
        } else {
          data = await askChef(text);
          const hasVideo = Boolean(data?.video_recommendation?.url);
          if (hasVideo) {
            stopAudioPlayback();
            els.voiceText.textContent = "영상을 띄웠습니다. 영상 확인 후 다시 말씀해주세요.";
          } else if (!data?.cancelled) {
            await speakText(data?.answer || "");
          }
        }
      } catch (error) {
        setStatus(error.message, true);
        addMessage("system", error.message);
        await speakText("처리하지 못했습니다.");
      } finally {
        state.processingVoice = false;
      }
    }

    init();
  </script>
</body>
</html>
        """
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "model_found": MODEL_PATH.exists(),
        "llm_base_url": LLM_BASE_URL,
    }


@app.get("/api/config")
async def get_config():
    return config_state()


@app.post("/api/config")
async def save_config(payload: SetupPayload):
    values = read_env_file(LLM_ENV_PATH)
    if payload.youtube_api_key is not None and payload.youtube_api_key.strip():
        values["YOUTUBE_API_KEY"] = payload.youtube_api_key.strip()
        os.environ["YOUTUBE_API_KEY"] = payload.youtube_api_key.strip()
    if payload.hf_token is not None and payload.hf_token.strip():
        token = payload.hf_token.strip()
        values["HF_TOKEN"] = token
        values["HUGGINGFACE_HUB_TOKEN"] = token
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    if payload.setup_complete:
        values["SETUP_COMPLETE"] = "1"
    write_env_file(LLM_ENV_PATH, values)
    return config_state()


@app.post("/api/restart-llm")
async def restart_llm():
    stopped = stop_managed_llm()
    ensure_llm_started()
    return {"status": "restarting", "stopped": stopped, "llm_process_alive": is_llm_process_alive()}


@app.get("/api/llm-health")
async def llm_health():
    stored_config = config_state()
    try:
        health = get_llm("/health", timeout=2)
        health["llm_autostart"] = AUTO_START_LLM
        health["llm_process_alive"] = is_llm_process_alive()
        health["youtube_key_configured"] = stored_config["youtube_api_key_configured"]
        health["youtube_enabled"] = bool(health.get("youtube_enabled") or stored_config["youtube_api_key_configured"])
        return health
    except HTTPException as exc:
        return {
            "status": "starting" if AUTO_START_LLM else "offline",
            "llm_loaded": False,
            "rag_loaded": False,
            "rag_mode": "none",
            "youtube_enabled": stored_config["youtube_api_key_configured"],
            "youtube_key_configured": stored_config["youtube_api_key_configured"],
            "llm_autostart": AUTO_START_LLM,
            "llm_process_alive": is_llm_process_alive(),
            "detail": exc.detail,
        }


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    frame = decode_image(image_bytes)
    return detect_ingredients(frame)


@app.post("/api/confirm")
async def confirm(payload: IngredientsPayload):
    return await run_in_threadpool(post_llm, "/vision", {"action": "confirm", "ingredients": payload.ingredients})


@app.post("/api/reject")
async def reject():
    return await run_in_threadpool(post_llm, "/vision", {"action": "reject", "ingredients": []})


@app.post("/api/ask")
async def ask(payload: AskPayload):
    return await run_in_threadpool(post_llm, "/ask", {"user_text": payload.user_text}, 90)


@app.post("/api/tts")
async def tts(payload: TtsPayload):
    audio, media_type = await run_in_threadpool(synthesize_tts_audio, payload.text)
    return Response(content=audio, media_type=media_type)


@app.post("/api/cancel")
async def cancel():
    return await run_in_threadpool(cancel_llm_output)


@app.post("/api/shutdown")
async def shutdown():
    threading.Thread(target=shutdown_processes, daemon=True).start()
    return {"status": "shutting_down"}


@app.get("/api/youtube-preview")
async def youtube_preview(query: str = ""):
    if not query.strip():
        return {
            "requested": False,
            "video_recommendation": None,
            "youtube_status": {"requested": False, "enabled": False, "message": ""},
        }
    return await run_in_threadpool(get_llm_with_params, "/youtube-preview", {"query": query}, 75)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("FRONT_HOST", "127.0.0.1")
    port = int(os.getenv("FRONT_PORT", "3000"))
    uvicorn.run(app, host=host, port=port)
