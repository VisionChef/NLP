import base64
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_CANDIDATES = (
    ROOT_DIR / "NLP-jm" / "Vision" / "best.pt",
    ROOT_DIR / "Vision" / "best.pt",
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
CONFIDENCE_THRESHOLD = float(os.getenv("VISION_CONFIDENCE", "0.4"))

app = FastAPI(title="Cooking Agent Frontend")
_model = None


class IngredientsPayload(BaseModel):
    ingredients: list[str] = Field(default_factory=list)


class AskPayload(BaseModel):
    user_text: str


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

        cv2.rectangle(frame, (x1, y1), (x2, y2), (38, 166, 91), 2)
        cv2.putText(
            frame,
            f"{name} {confidence:.2f}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (38, 166, 91),
            2,
        )

    ingredients = sorted({item["name"] for item in detections})
    ok, encoded = cv2.imencode(".jpg", frame)
    preview = base64.b64encode(encoded.tobytes()).decode("ascii") if ok else None
    return {"ingredients": ingredients, "detections": detections, "preview": preview}


def post_llm(path: str, payload: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    try:
        response = requests.post(f"{LLM_BASE_URL}{path}", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"LLM server request failed: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="LLM server returned invalid JSON.") from exc


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(
        """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Cooking Agent</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f4;
      --panel: #ffffff;
      --ink: #202421;
      --muted: #66706a;
      --line: #dfe4dc;
      --accent: #268653;
      --accent-dark: #1b6840;
      --warn: #b65c20;
      --danger: #b23b3b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, "Noto Sans KR", sans-serif;
      background: var(--bg);
      color: var(--ink);
      letter-spacing: 0;
    }
    header {
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 24px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfa;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
    }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 22px;
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(320px, 0.8fr);
      gap: 18px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .toolbar {
      display: flex;
      gap: 8px;
      padding: 14px;
      border-bottom: 1px solid var(--line);
      align-items: center;
      flex-wrap: wrap;
    }
    .segmented {
      display: inline-grid;
      grid-template-columns: 1fr 1fr;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #eef2ed;
    }
    button {
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      padding: 0 14px;
      font-size: 14px;
      cursor: pointer;
      white-space: nowrap;
    }
    .segmented button {
      border: 0;
      border-radius: 0;
      background: transparent;
    }
    button.active, button.primary {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }
    button.primary:hover { background: var(--accent-dark); }
    button.danger {
      color: var(--danger);
      border-color: #e3c7c7;
    }
    input[type="file"], input[type="text"] {
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      background: #fff;
      font-size: 14px;
    }
    .workspace {
      padding: 14px;
      display: grid;
      gap: 12px;
    }
    .mode-panel[hidden] { display: none; }
    .camera-grid {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: end;
    }
    video, .preview {
      width: 100%;
      aspect-ratio: 16 / 10;
      background: #151a17;
      object-fit: contain;
      border-radius: 8px;
      border: 1px solid var(--line);
    }
    .preview.empty {
      display: grid;
      place-items: center;
      color: var(--muted);
      font-size: 14px;
      background: #f1f3ef;
    }
    .side {
      display: grid;
      grid-template-rows: auto 1fr;
      min-height: 620px;
    }
    .results {
      padding: 14px;
      display: grid;
      align-content: start;
      gap: 12px;
    }
    .ingredients {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      min-height: 40px;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      min-height: 30px;
      border: 1px solid #cfe2d5;
      background: #eef8f1;
      border-radius: 999px;
      padding: 0 10px;
      font-size: 14px;
    }
    .detections {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 13px;
    }
    .status {
      min-height: 22px;
      color: var(--muted);
      font-size: 13px;
    }
    .status.error { color: var(--danger); }
    .answer {
      border-top: 1px solid var(--line);
      padding-top: 12px;
      display: grid;
      gap: 10px;
    }
    .answer-box {
      min-height: 100px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fafbf9;
      font-size: 14px;
      line-height: 1.55;
      white-space: pre-wrap;
    }
    .ask-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
    }
    @media (max-width: 820px) {
      header { padding: 0 14px; }
      main {
        grid-template-columns: 1fr;
        padding: 14px;
      }
      .side { min-height: auto; }
      .camera-grid, .ask-row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Cooking Agent</h1>
    <div id="serverState" class="status"></div>
  </header>
  <main>
    <section>
      <div class="toolbar">
        <div class="segmented" role="tablist" aria-label="input mode">
          <button id="cameraTab" class="active" type="button" data-mode="camera">카메라</button>
          <button id="uploadTab" type="button" data-mode="upload">업로드</button>
        </div>
        <button id="detectButton" class="primary" type="button">인식</button>
      </div>
      <div class="workspace">
        <div id="cameraPanel" class="mode-panel">
          <div class="camera-grid">
            <video id="camera" autoplay playsinline muted></video>
            <button id="startCamera" type="button">카메라 켜기</button>
          </div>
        </div>
        <div id="uploadPanel" class="mode-panel" hidden>
          <input id="imageFile" type="file" accept="image/*" />
        </div>
        <img id="previewImage" class="preview" hidden alt="detected preview" />
        <div id="emptyPreview" class="preview empty">이미지 없음</div>
        <div id="status" class="status"></div>
      </div>
    </section>
    <section class="side">
      <div class="toolbar">
        <button id="confirmButton" class="primary" type="button">확정</button>
        <button id="clearButton" class="danger" type="button">초기화</button>
      </div>
      <div class="results">
        <div id="ingredients" class="ingredients"></div>
        <div id="detections" class="detections"></div>
        <div class="answer">
          <div class="ask-row">
            <input id="askText" type="text" placeholder="셰프에게 물어보기" />
            <button id="askButton" type="button">전송</button>
          </div>
          <div id="answerBox" class="answer-box"></div>
        </div>
      </div>
    </section>
  </main>
  <canvas id="captureCanvas" hidden></canvas>
  <script>
    const state = { mode: "camera", stream: null, ingredients: [] };
    const els = {
      camera: document.getElementById("camera"),
      cameraPanel: document.getElementById("cameraPanel"),
      uploadPanel: document.getElementById("uploadPanel"),
      imageFile: document.getElementById("imageFile"),
      previewImage: document.getElementById("previewImage"),
      emptyPreview: document.getElementById("emptyPreview"),
      status: document.getElementById("status"),
      serverState: document.getElementById("serverState"),
      ingredients: document.getElementById("ingredients"),
      detections: document.getElementById("detections"),
      answerBox: document.getElementById("answerBox"),
      askText: document.getElementById("askText"),
      canvas: document.getElementById("captureCanvas")
    };

    function setStatus(text, error = false) {
      els.status.textContent = text;
      els.status.classList.toggle("error", error);
    }

    function setMode(mode) {
      state.mode = mode;
      document.querySelectorAll("[data-mode]").forEach((button) => {
        button.classList.toggle("active", button.dataset.mode === mode);
      });
      els.cameraPanel.hidden = mode !== "camera";
      els.uploadPanel.hidden = mode !== "upload";
      setStatus("");
    }

    async function startCamera() {
      if (state.stream) return;
      try {
        state.stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment" },
          audio: false
        });
        els.camera.srcObject = state.stream;
        setStatus("카메라 준비됨");
      } catch (error) {
        setStatus(`카메라를 열 수 없습니다: ${error.message}`, true);
      }
    }

    function renderDetection(data) {
      state.ingredients = data.ingredients || [];
      els.ingredients.innerHTML = state.ingredients.length
        ? state.ingredients.map((item) => `<span class="chip">${item}</span>`).join("")
        : `<span class="status">인식된 재료 없음</span>`;
      els.detections.innerHTML = (data.detections || [])
        .map((item) => `${item.name} · ${(item.confidence * 100).toFixed(1)}%`)
        .join("<br>");
      if (data.preview) {
        els.previewImage.src = `data:image/jpeg;base64,${data.preview}`;
        els.previewImage.hidden = false;
        els.emptyPreview.hidden = true;
      }
    }

    async function detectBlob(blob, filename) {
      const formData = new FormData();
      formData.append("file", blob, filename);
      setStatus("인식 중...");
      const response = await fetch("/api/detect", { method: "POST", body: formData });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "인식 실패");
      renderDetection(data);
      setStatus(`${state.ingredients.length}개 재료 인식됨`);
    }

    async function detectFromCamera() {
      if (!state.stream) await startCamera();
      if (!state.stream) return;
      const video = els.camera;
      const canvas = els.canvas;
      canvas.width = video.videoWidth || 1280;
      canvas.height = video.videoHeight || 720;
      canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
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

    async function confirmIngredients() {
      if (!state.ingredients.length) {
        setStatus("확정할 재료가 없습니다.", true);
        return;
      }
      setStatus("LLM 서버로 전송 중...");
      const response = await fetch("/api/confirm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ingredients: state.ingredients })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "전송 실패");
      setStatus(`확정됨 · 레시피 ${data.recipes_found ?? 0}개`);
    }

    async function askChef() {
      const text = els.askText.value.trim();
      if (!text) return;
      els.answerBox.textContent = "답변 생성 중...";
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_text: text })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "질문 실패");
      els.answerBox.textContent = data.answer || "";
      if (data.video_recommendation?.url) {
        els.answerBox.textContent += `\n\nYouTube: ${data.video_recommendation.url}`;
      }
    }

    function clearAll() {
      state.ingredients = [];
      els.ingredients.innerHTML = "";
      els.detections.innerHTML = "";
      els.answerBox.textContent = "";
      els.previewImage.hidden = true;
      els.previewImage.removeAttribute("src");
      els.emptyPreview.hidden = false;
      setStatus("");
    }

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
      try { await confirmIngredients(); } catch (error) { setStatus(error.message, true); }
    });
    document.getElementById("askButton").addEventListener("click", async () => {
      try { await askChef(); } catch (error) { els.answerBox.textContent = error.message; }
    });
    document.getElementById("clearButton").addEventListener("click", clearAll);
    els.serverState.textContent = "Front ready";
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


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    frame = decode_image(image_bytes)
    return detect_ingredients(frame)


@app.post("/api/confirm")
async def confirm(payload: IngredientsPayload):
    return post_llm("/vision", {"action": "confirm", "ingredients": payload.ingredients})


@app.post("/api/ask")
async def ask(payload: AskPayload):
    return post_llm("/ask", {"user_text": payload.user_text}, timeout=90)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("FRONT_HOST", "127.0.0.1")
    port = int(os.getenv("FRONT_PORT", "3000"))
    uvicorn.run(app, host=host, port=port)
