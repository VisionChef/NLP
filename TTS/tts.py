import os
import re
from io import BytesIO


TTS_PROVIDER = os.getenv("TTS_PROVIDER", "gtts").strip().lower()
VARCO_TTS_URL = os.getenv("VARCO_TTS_URL", "").strip()
VARCO_API_KEY = os.getenv("VARCO_API_KEY", "").strip()


def clean_tts_text(text: str) -> str:
    clean = text or ""
    clean = re.sub(r"(\d+)\s*[-~]\s*(\d+)", r"\1에서 \2", clean)
    clean = re.sub(r"[^\w\s가-힣?.!,]", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:900]


def synthesize_gtts(text: str) -> tuple[bytes, str]:
    from gtts import gTTS
    fp = BytesIO()
    gTTS(text=text, lang="ko", slow=False, lang_check=False).write_to_fp(fp)
    return fp.getvalue(), "audio/mpeg"


def synthesize_varco_tts(text: str) -> tuple[bytes, str]:
    import requests
    response = requests.post(
        VARCO_TTS_URL,
        headers={
            "Authorization": f"Bearer {VARCO_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"text": text, "language": "ko-KR"},
        timeout=30,
    )
    response.raise_for_status()
    return response.content, response.headers.get("content-type", "audio/mpeg")


def synthesize_tts_audio(text: str) -> tuple[bytes, str]:
    clean = clean_tts_text(text)
    if not clean:
        raise ValueError("TTS로 읽을 텍스트가 없습니다.")
    if TTS_PROVIDER == "varco":
        return synthesize_varco_tts(clean)
    return synthesize_gtts(clean)
