import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import requests
from requests import RequestException

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_WATCH_URL = "https://www.youtube.com/watch?v="
YOUTUBE_EMBED_URL = "https://www.youtube.com/embed/"
MODULE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(MODULE_DIR / "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_EXPERIMENTAL_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
_WHISPER_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}
_LAST_YOUTUBE_ERROR = ""


def get_last_youtube_error() -> str:
    return _LAST_YOUTUBE_ERROR


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class YouTubeRecommendationConfig:
    max_results: int = 2
    region_code: str = "KR"
    relevance_language: str = "ko"
    transcript_languages: tuple[str, ...] = ("ko", "en")
    top_segments: int = 3
    segment_window_size: int = 6
    segment_stride: int = 3
    min_segment_score: float = 0.03
    timeout_seconds: int = 15
    allow_metadata_fallback: bool = False
    enable_asr_fallback: bool = True
    asr_max_videos: int = 2
    asr_model_name: str = "small"
    asr_language: str = "ko"
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    asr_cache_dir: str = str(MODULE_DIR / "youtube_transcript_cache")
    asr_model_cache_dir: str = str(MODULE_DIR / "youtube_model_cache")


DEFAULT_CONFIG = YouTubeRecommendationConfig(
    allow_metadata_fallback=_env_flag("YOUTUBE_METADATA_FALLBACK", True),
    enable_asr_fallback=_env_flag("YOUTUBE_ASR_FALLBACK", False),
    asr_max_videos=_env_int("YOUTUBE_ASR_MAX_VIDEOS", 2),
    asr_model_name=os.getenv("YOUTUBE_ASR_MODEL", "small"),
    asr_language=os.getenv("YOUTUBE_ASR_LANGUAGE", "ko"),
    asr_device=os.getenv("YOUTUBE_ASR_DEVICE", "cpu"),
    asr_compute_type=os.getenv("YOUTUBE_ASR_COMPUTE_TYPE", "int8"),
    asr_cache_dir=os.getenv("YOUTUBE_ASR_CACHE_DIR", str(MODULE_DIR / "youtube_transcript_cache")),
    asr_model_cache_dir=os.getenv("YOUTUBE_ASR_MODEL_CACHE_DIR", str(MODULE_DIR / "youtube_model_cache")),
)


def normalize_text_for_tfidf(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^0-9a-z가-힣 ]+", " ", text)
    return text.strip()


def _char_ngrams(text: str, n: int = 2) -> set[str]:
    compact = text.replace(" ", "")
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[index : index + n] for index in range(len(compact) - n + 1)}


def _basic_similarity(document: str, query: str) -> float:
    document_tokens = set(document.split())
    query_tokens = set(query.split())
    token_score = 0.0
    if query_tokens:
        token_score = len(document_tokens & query_tokens) / len(query_tokens)

    document_grams = _char_ngrams(document, n=2)
    query_grams = _char_ngrams(query, n=2)
    gram_score = 0.0
    if query_grams:
        gram_score = len(document_grams & query_grams) / len(query_grams)

    return max(token_score, gram_score)


def _similarity_scores(documents: list[str], query_text: str) -> list[float]:
    if not documents:
        return []

    if TfidfVectorizer is None or cosine_similarity is None:
        return [_basic_similarity(document, query_text) for document in documents]

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    try:
        tfidf_matrix = vectorizer.fit_transform(documents + [query_text])
    except ValueError:
        return [0.0 for _ in documents]

    query_vector = tfidf_matrix[-1]
    document_matrix = tfidf_matrix[:-1]
    return [float(score) for score in cosine_similarity(document_matrix, query_vector).ravel()]


def build_youtube_search_query(query: str) -> str:
    cleaned = normalize_text_for_tfidf(query)
    cleaned = re.sub(r"([가-힣]+)(써는|자르는|만드는|하는|볶는|삶는|끓이는|굽는|튀기는)법", r"\1 \2 법", cleaned)
    filler_patterns = [
        r"\b모르겠(?:어|어요|습니다|다|는데)?\b",
        r"\b알려\s*줘(?:요)?\b",
        r"\b보여\s*줘(?:요)?\b",
        r"\b틀어\s*줘(?:요)?\b",
        r"\b유튜브\b",
        r"\b영상\b",
        r"\b동영상\b",
    ]
    for pattern in filler_patterns:
        cleaned = re.sub(pattern, " ", cleaned)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        cleaned = normalize_text_for_tfidf(query)

    compact = cleaned.replace(" ", "")
    cooking_terms = (
        "요리",
        "레시피",
        "손질",
        "썰",
        "자르",
        "다지",
        "볶",
        "삶",
        "끓",
        "굽",
        "튀기",
        "까",
        "벗기",
        "씻",
        "익히",
    )
    suffix = "요리 방법" if not any(term in compact for term in cooking_terms) else "요리"
    return f"{cleaned} {suffix}".strip()


def build_watch_url(video_id: str, start_seconds: Optional[Union[int, float]] = None) -> str:
    if start_seconds is None:
        return f"{YOUTUBE_WATCH_URL}{video_id}"

    start = max(0, int(start_seconds))
    return f"{YOUTUBE_WATCH_URL}{video_id}&t={start}s"


def build_embed_url(video_id: str, start_seconds: Optional[Union[int, float]] = None) -> str:
    if start_seconds is None:
        return f"{YOUTUBE_EMBED_URL}{video_id}"

    start = max(0, int(start_seconds))
    return f"{YOUTUBE_EMBED_URL}{video_id}?start={start}&autoplay=1"


def search_youtube_videos(
    query: str,
    api_key: str,
    config: YouTubeRecommendationConfig = DEFAULT_CONFIG,
) -> list[dict[str, Any]]:
    global _LAST_YOUTUBE_ERROR
    _LAST_YOUTUBE_ERROR = ""

    if not api_key:
        return []

    search_query = build_youtube_search_query(query)
    params = {
        "part": "snippet",
        "q": search_query,
        "type": "video",
        "maxResults": config.max_results,
        "relevanceLanguage": config.relevance_language,
        "regionCode": config.region_code,
        "safeSearch": "moderate",
        "videoEmbeddable": "true",
        "key": api_key,
    }

    try:
        response = requests.get(YOUTUBE_SEARCH_URL, params=params, timeout=config.timeout_seconds)
        response.raise_for_status()
        data = response.json()
    except RequestException as exc:
        response = getattr(exc, "response", None)
        if response is not None:
            _LAST_YOUTUBE_ERROR = f"{response.status_code}: {response.text[:300]}"
        else:
            _LAST_YOUTUBE_ERROR = str(exc)
        return []
    except ValueError as exc:
        _LAST_YOUTUBE_ERROR = f"Invalid JSON response: {exc}"
        return []

    videos = []
    for index, item in enumerate(data.get("items", [])):
        video_id = item.get("id", {}).get("videoId")
        snippet = item.get("snippet", {})
        if not video_id:
            continue

        videos.append(
            {
                "video_id": video_id,
                "rank": index + 1,
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "channel_title": snippet.get("channelTitle", ""),
                "published_at": snippet.get("publishedAt", ""),
                "thumbnail": snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
                "url": build_watch_url(video_id),
                "embed_url": build_embed_url(video_id),
                "search_query": search_query,
            }
        )

    return videos


def _snippet_to_dict(snippet: Any) -> dict[str, Any]:
    if isinstance(snippet, dict):
        return {
            "text": snippet.get("text", ""),
            "start": float(snippet.get("start", 0.0)),
            "duration": float(snippet.get("duration", 0.0)),
        }

    return {
        "text": getattr(snippet, "text", ""),
        "start": float(getattr(snippet, "start", 0.0)),
        "duration": float(getattr(snippet, "duration", 0.0)),
    }


def _transcript_to_raw_data(transcript: Any) -> list[dict[str, Any]]:
    if hasattr(transcript, "to_raw_data"):
        return [_snippet_to_dict(snippet) for snippet in transcript.to_raw_data()]

    return [_snippet_to_dict(snippet) for snippet in transcript]


def get_video_transcript(
    video_id: str,
    languages: Optional[Union[tuple[str, ...], list[str]]] = None,
) -> list[dict[str, Any]]:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ModuleNotFoundError:
        return []

    language_priority = list(languages or DEFAULT_CONFIG.transcript_languages)

    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages=language_priority)
        return _transcript_to_raw_data(transcript)
    except Exception:
        pass

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=language_priority)
        return _transcript_to_raw_data(transcript)
    except Exception:
        return []


def _asr_cache_path(video_id: str, config: YouTubeRecommendationConfig) -> Path:
    safe_video_id = re.sub(r"[^0-9A-Za-z_-]", "", video_id)
    return Path(config.asr_cache_dir) / f"{safe_video_id}.json"


def _load_asr_transcript_cache(video_id: str, config: YouTubeRecommendationConfig) -> list[dict[str, Any]]:
    cache_path = _asr_cache_path(video_id, config)
    if not cache_path.exists():
        return []

    try:
        with cache_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []

    transcript = data.get("transcript", [])
    if not isinstance(transcript, list):
        return []
    return transcript


def _save_asr_transcript_cache(
    video_id: str,
    transcript: list[dict[str, Any]],
    config: YouTubeRecommendationConfig,
) -> None:
    cache_path = _asr_cache_path(video_id, config)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "video_id": video_id,
                    "model": config.asr_model_name,
                    "language": config.asr_language,
                    "transcript": transcript,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except OSError:
        return


def _download_audio(video_id: str, output_dir: Path) -> Optional[Path]:
    try:
        import yt_dlp
    except ImportError:
        return None

    output_template = str(output_dir / f"{video_id}.%(ext)s")
    options = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(build_watch_url(video_id), download=True)
            filename = Path(ydl.prepare_filename(info))
    except Exception:
        return None

    if filename.exists():
        return filename

    matches = list(output_dir.glob(f"{video_id}.*"))
    return matches[0] if matches else None


def _get_whisper_model(config: YouTubeRecommendationConfig) -> Optional[Any]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None

    key = (
        config.asr_model_name,
        config.asr_device,
        config.asr_compute_type,
        config.asr_model_cache_dir,
    )
    if key not in _WHISPER_MODEL_CACHE:
        try:
            Path(config.asr_model_cache_dir).mkdir(parents=True, exist_ok=True)
            _WHISPER_MODEL_CACHE[key] = WhisperModel(
                config.asr_model_name,
                device=config.asr_device,
                compute_type=config.asr_compute_type,
                download_root=config.asr_model_cache_dir,
            )
        except Exception as exc:
            print(f"⚠️ [YouTube ASR] Whisper 모델 로드/다운로드 실패: {exc}")
            return None

    return _WHISPER_MODEL_CACHE[key]


def transcribe_video_with_asr(
    video_id: str,
    config: YouTubeRecommendationConfig = DEFAULT_CONFIG,
) -> list[dict[str, Any]]:
    cached_transcript = _load_asr_transcript_cache(video_id, config)
    if cached_transcript:
        return cached_transcript

    model = _get_whisper_model(config)
    if model is None:
        return []

    cache_dir = Path(config.asr_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="youtube_audio_", dir=str(cache_dir)) as tmp_dir:
        audio_path = _download_audio(video_id, Path(tmp_dir))
        if not audio_path:
            return []

        try:
            segments, _ = model.transcribe(
                str(audio_path),
                language=config.asr_language or None,
                vad_filter=True,
            )
            transcript = []
            for segment in segments:
                text = getattr(segment, "text", "").strip()
                if not text:
                    continue

                start = float(getattr(segment, "start", 0.0))
                end = float(getattr(segment, "end", start))
                transcript.append(
                    {
                        "text": text,
                        "start": start,
                        "duration": max(0.0, end - start),
                    }
                )
        except Exception:
            return []

    if transcript:
        _save_asr_transcript_cache(video_id, transcript, config)

    return transcript


def build_transcript_segments(
    transcript: list[dict[str, Any]],
    window_size: int = DEFAULT_CONFIG.segment_window_size,
    stride: int = DEFAULT_CONFIG.segment_stride,
) -> list[dict[str, Any]]:
    if not transcript:
        return []

    window_size = max(1, window_size)
    stride = max(1, stride)
    segments = []

    for start_index in range(0, len(transcript), stride):
        chunk = transcript[start_index : start_index + window_size]
        if not chunk:
            break

        raw_text = " ".join(item.get("text", "") for item in chunk).strip()
        text = normalize_text_for_tfidf(raw_text)
        if not text:
            continue

        start = float(chunk[0].get("start", 0.0))
        end = float(chunk[-1].get("start", 0.0)) + float(chunk[-1].get("duration", 0.0))
        segments.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "start_seconds": max(0, int(start)),
                "text": text,
                "raw_text": raw_text,
            }
        )

        if start_index + window_size >= len(transcript):
            break

    return segments


def rank_transcript_segments(
    query: str,
    transcript: list[dict[str, Any]],
    top_k: int = DEFAULT_CONFIG.top_segments,
    config: YouTubeRecommendationConfig = DEFAULT_CONFIG,
) -> list[dict[str, Any]]:
    segments = build_transcript_segments(
        transcript,
        window_size=config.segment_window_size,
        stride=config.segment_stride,
    )
    if not segments:
        return []

    query_text = normalize_text_for_tfidf(f"{query} {build_youtube_search_query(query)}")
    documents = [segment["text"] for segment in segments]

    scores = _similarity_scores(documents, query_text)

    for segment, score in zip(segments, scores):
        segment["score"] = float(round(score, 4))

    ranked = sorted(segments, key=lambda item: item["score"], reverse=True)
    return [segment for segment in ranked if segment["score"] >= config.min_segment_score][:top_k]


def _attach_segment_urls(video: dict[str, Any], segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    video_id = video["video_id"]
    enriched_segments = []
    for segment in segments:
        start_seconds = segment["start_seconds"]
        enriched_segments.append(
            {
                **segment,
                "url": build_watch_url(video_id, start_seconds),
                "embed_url": build_embed_url(video_id, start_seconds),
            }
        )
    return enriched_segments


def _metadata_fallback_score(video: dict[str, Any], query: str) -> float:
    text = normalize_text_for_tfidf(
        f"{video.get('title', '')} {video.get('description', '')} {video.get('channel_title', '')}"
    )
    query_text = normalize_text_for_tfidf(f"{query} {build_youtube_search_query(query)}")
    if not text or not query_text:
        return 0.0

    return _similarity_scores([text], query_text)[0]


def _build_candidate(
    video: dict[str, Any],
    best_segments: list[dict[str, Any]],
    match_source: str,
    config: YouTubeRecommendationConfig,
) -> dict[str, Any]:
    best_segments = _attach_segment_urls(video, best_segments)
    top_score = best_segments[0]["score"]
    rank_bonus = max(0.0, (config.max_results - video["rank"]) * 0.01)
    return {
        **video,
        "url": best_segments[0]["url"],
        "embed_url": best_segments[0]["embed_url"],
        "start_seconds": best_segments[0]["start_seconds"],
        "best_segments": best_segments,
        "match_score": round(top_score + rank_bonus, 4),
        "match_source": match_source,
        "timeline_found": True,
    }


def find_best_youtube_segment(
    query: str,
    api_key: str,
    config: YouTubeRecommendationConfig = DEFAULT_CONFIG,
) -> Optional[dict[str, Any]]:
    if not api_key:
        return None

    videos = search_youtube_videos(query, api_key, config=config)
    if not videos:
        return None

    candidates = []
    fallbacks = []
    asr_budget = max(0, config.asr_max_videos)

    for video in videos:
        transcript = get_video_transcript(video["video_id"], languages=config.transcript_languages)
        match_source = "transcript"

        if not transcript and config.enable_asr_fallback and asr_budget > 0:
            transcript = transcribe_video_with_asr(video["video_id"], config=config)
            match_source = "asr"
            asr_budget -= 1

        best_segments = rank_transcript_segments(query, transcript, top_k=config.top_segments, config=config)
        if best_segments:
            candidates.append(_build_candidate(video, best_segments, match_source, config))
            continue

        if config.allow_metadata_fallback:
            fallback_score = _metadata_fallback_score(video, query)
            fallbacks.append(
                {
                    **video,
                    "start_seconds": 0,
                    "best_segments": [],
                    "match_score": round(fallback_score, 4),
                    "match_source": "metadata",
                    "timeline_found": False,
                    "reason": "No transcript/ASR segment matched. This is a full-video fallback.",
                }
            )

    if candidates:
        return max(candidates, key=lambda item: item["match_score"])

    if config.allow_metadata_fallback and fallbacks:
        return max(fallbacks, key=lambda item: item["match_score"])

    return None


def is_cooking_video_query(text: str) -> bool:
    lower = text.strip().lower()
    compact = re.sub(r"\s+", "", lower)

    explicit_video_intents = [
        "유튜브",
        "영상",
        "동영상",
        "보여줘",
        "보여주세요",
        "틀어줘",
        "틀어주세요",
        "나와",
        "나오는",
        "찾아줘",
        "추천해줘",
        "시연",
        "demo",
        "recipe",
    ]
    cooking_method_terms = [
        "자르는법",
        "써는법",
        "만드는법",
        "하는법",
        "손질",
        "썰기",
        "썰어",
        "자르",
        "다지",
        "볶는법",
        "삶는법",
        "끓이는법",
        "불리는법",
        "불리는",
        "불리",
        "굽는법",
        "튀기는법",
        "까는법",
        "벗기는법",
        "씻는법",
        "익히는법",
        "레시피",
        "요리방법",
    ]

    return any(term in lower for term in explicit_video_intents) or any(
        term in compact for term in cooking_method_terms
    )
