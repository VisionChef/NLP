import os
import re
import sys
from getpass import getpass
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional
import tempfile
import threading

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

# ⚠️ Hugging Face / Transformers 캐시는 관련 라이브러리 import 전에 잡아둔다.
DEFAULT_LOCAL_MODEL_DIR = r"D:\models\skt_A.X-4.0-Light"
DEFAULT_HF_HOME = r"D:\models\hf_cache"
os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
os.environ.setdefault("HF_HUB_DISABLE_EXPERIMENTAL_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, pipeline
from huggingface_hub import login, snapshot_download
from gtts import gTTS
import pygame
import time
from starlette.concurrency import run_in_threadpool

from Rag.rag import (
    load_recipes, 
    build_vectorstore, 
    search_recipes, 
    normalize_ingredient,
    BOOK_RECIPES_FILE, 
    TRENDING_RECIPES_FILE, 
    CHROMA_DIR
)
from youtube_api import find_best_youtube_segment, get_last_youtube_error, is_cooking_video_query

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent


def read_env_file(path: Path) -> dict[str, str]:
    values = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        name, value = line.split("=", 1)
        values[name.strip()] = value.strip().strip('"').strip("'")
    return values


def get_runtime_env(name: str) -> Optional[str]:
    return (
        os.getenv(name)
        or read_env_file(MODULE_DIR / ".env").get(name)
        or read_env_file(PROJECT_DIR / ".env").get(name)
    )

# ==========================================
# ⚙️ 설정, 모델 로드, RAG 초기화
# ==========================================
pipe = None
vectorstore = None # RAG 벡터 저장소
recipe_documents = []
loaded_model_source = None
loaded_quantization = "none"
rag_error = None
rag_mode = "none"
tts_lock = threading.Lock()
generation_lock = threading.Lock()
generation_state_lock = threading.Lock()
generation_cancel_event = threading.Event()
generation_active = False
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "skt/A.X-4.0-Light")
LLM_LOCAL_MODEL_DIR = os.getenv("LLM_LOCAL_MODEL_DIR", DEFAULT_LOCAL_MODEL_DIR)
LLM_LOAD_IN_8BIT = os.getenv("LLM_LOAD_IN_8BIT", "0").strip().lower() in {"1", "true", "yes", "on"}
SYSTEM_PROMPT = """너는 사용자 옆에서 같이 요리하는 만능 셰프야.
말투는 사람과 대화하듯 자연스럽고 친근하게 해. 사용자를 가르치는 설명서가 아니라, 지금 주방에서 같이 조리하는 셰프처럼 반응해.
재료 손질, 조리 순서, 대체 재료, 간 맞추기, 실패 수습, 보관법, 플레이팅까지 폭넓게 도와줘.
사용자의 말이 짧거나 애매하면 먼저 상황을 짚고, 필요한 질문은 한 가지만 물어봐.
요리 실행을 안내할 때는 반드시 아래 방식을 지켜.
지금 사용자가 바로 실행할 한 단계만 말해. 전체 레시피, 전체 순서, 다음 단계 목록을 한 번에 말하지 마.
한 단계 안에서는 아주 구체적으로 말해. 양, 불 세기, 시간, 손의 움직임, 냄새와 색 변화, 익은 상태 기준, 흔한 실수와 주의점을 포함해.
답변은 3문장 이상 6문장 이하로 자연스럽게 말해. 너무 짧게 끝내지 말고, 사용자가 바로 따라 할 수 있을 만큼 디테일하게 말해.
사용자가 "다 했어", "다음", "계속", "했어"처럼 진행 신호를 주면 그때 다음 단계로 넘어가.
사용자가 전체 레시피를 물어도 전체를 나열하지 말고, 먼저 시작 단계부터 같이 진행해.
절대 *, #, - 같은 기호나 번호 목록을 쓰지 말고 구어체로만 답해.
문서 내용과 관련된 질문이면 아래 참고 문서 내용을 바탕으로 답하되, 사용자의 현재 조리 상황과 대화를 우선해.
---
[참고 문서 내용]
{rag_context}
---
"""


class GenerationCancelled(Exception):
    pass


class CancelStoppingCriteria(StoppingCriteria):
    def __init__(self, cancel_event: threading.Event):
        self.cancel_event = cancel_event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.cancel_event.is_set()


def set_generation_active(active: bool) -> None:
    global generation_active
    with generation_state_lock:
        generation_active = active


def is_generation_active() -> bool:
    with generation_state_lock:
        return generation_active


def generate_llm_answer(prompt: str) -> str:
    if pipe is None:
        raise RuntimeError("LLM model is not loaded yet.")

    if not generation_lock.acquire(blocking=False):
        raise RuntimeError("이미 답변을 생성하는 중입니다. 먼저 중단하거나 잠시 기다려주세요.")

    generation_cancel_event.clear()
    set_generation_active(True)
    try:
        outputs = pipe(
            prompt,
            max_new_tokens=360,
            do_sample=True,
            temperature=0.62,
            repetition_penalty=1.08,
            eos_token_id=pipe.tokenizer.eos_token_id,
            pad_token_id=pipe.tokenizer.pad_token_id,
            stopping_criteria=StoppingCriteriaList([CancelStoppingCriteria(generation_cancel_event)]),
        )
        if generation_cancel_event.is_set():
            raise GenerationCancelled()
        full_text = outputs[0]["generated_text"]
        return full_text.split("<|im_start|>assistant\n")[-1].strip()
    finally:
        set_generation_active(False)
        generation_lock.release()


def has_local_model(model_dir: str) -> bool:
    path = Path(model_dir)
    if not path.exists() or not path.is_dir():
        return False

    has_config = (path / "config.json").exists()
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
    return has_config and has_weights


def get_hf_token(required: bool) -> Optional[str]:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token and required:
        token = getpass("Hugging Face token을 입력하세요: ").strip()

    if required and not token:
        raise ValueError("로컬 모델이 없어 다운로드가 필요합니다. HF_TOKEN을 입력해주세요.")

    if token:
        login(token=token, add_to_git_credential=False)
        os.environ["HF_TOKEN"] = token

    return token


def resolve_model_source() -> str:
    local_model_dir = os.getenv("LLM_LOCAL_MODEL_DIR", LLM_LOCAL_MODEL_DIR)
    if has_local_model(local_model_dir):
        print(f"📦 로컬 A.X 모델 사용: {local_model_dir}")
        get_hf_token(required=False)
        return local_model_dir

    print(f"📦 로컬 모델 없음: {local_model_dir}")
    print(f"⬇️ Hugging Face에서 {LLM_MODEL_ID} 다운로드를 시작합니다.")
    token = get_hf_token(required=True)
    Path(local_model_dir).mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=LLM_MODEL_ID,
        local_dir=local_model_dir,
        token=token,
    )
    print(f"✅ 모델 다운로드 완료: {local_model_dir}")
    return local_model_dir


def build_quantization_config():
    global loaded_quantization

    if not LLM_LOAD_IN_8BIT:
        loaded_quantization = "none"
        return None

    if not torch.cuda.is_available():
        print("⚠️ 8bit 양자화는 CUDA 환경에서만 사용하도록 설정했습니다. 일반 로드로 전환합니다.")
        loaded_quantization = "none"
        return None

    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        print("⚠️ bitsandbytes가 없어 8bit 양자화를 적용하지 못했습니다. 일반 로드로 전환합니다.")
        loaded_quantization = "none"
        return None

    loaded_quantization = "8bit"
    return BitsAndBytesConfig(load_in_8bit=True)


def load_llm_pipeline(model_source: str):
    quantization_config = build_quantization_config()
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=True,
    )
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        **model_kwargs,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 실행
    global pipe, vectorstore, recipe_documents, loaded_model_source, rag_error, rag_mode
    print("🚀 서버 시작...")
    
    # LLM 모델 로드
    loaded_model_source = resolve_model_source()
    print(f"🧠 A.X 로딩 중... source={loaded_model_source}")
    pipe = load_llm_pipeline(loaded_model_source)
    print(f"✅ A.X 모델 준비 완료! quantization={loaded_quantization}")

    # RAG 벡터 저장소 로드 또는 생성
    # rag.py가 main.py와 같은 LLM 폴더 아래에 있으므로, rag.py 내부의 상대경로가 잘 동작함
    rag_dir = os.path.join(os.path.dirname(__file__), "Rag")
    book_recipes_path = os.path.join(rag_dir, BOOK_RECIPES_FILE)
    trending_recipes_path = os.path.join(rag_dir, TRENDING_RECIPES_FILE)
    chroma_path = os.path.join(rag_dir, CHROMA_DIR)

    book_docs = load_recipes(book_recipes_path, source_type="Baek_Book")
    trending_docs = load_recipes(trending_recipes_path, source_type="trending")
    recipe_documents = book_docs + trending_docs

    try:
        vectorstore = build_vectorstore(recipe_documents, chroma_path)
        rag_error = None
        rag_mode = "vector"
        print("✅ RAG 벡터 검색 준비 완료")
    except Exception as exc:
        vectorstore = None
        rag_error = str(exc)
        rag_mode = "fallback" if recipe_documents else "none"
        print(f"⚠️ RAG 벡터 초기화 실패. JSON fallback 검색으로 계속 실행합니다: {exc}")

    yield
    # 서버 종료 시 실행 (여기서는 특별한 정리 작업 없음)
    print("🌙 서버 종료...")

app = FastAPI(lifespan=lifespan)

current_ingredients = []
pending_ingredients = []
cached_rag_context = "없음"
chat_history = [] 


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "llm_loaded": pipe is not None,
        "rag_loaded": vectorstore is not None or bool(recipe_documents),
        "rag_mode": rag_mode,
        "rag_error": rag_error,
        "youtube_enabled": bool(get_runtime_env("YOUTUBE_API_KEY")),
        "llm_generating": is_generation_active(),
        "model_source": loaded_model_source,
        "quantization": loaded_quantization,
        "current_ingredients": current_ingredients,
        "pending_ingredients": pending_ingredients,
    }

# ==========================================
# 🔊 TTS 재생 함수
# ==========================================
def play_tts(text: str):
    clean_text = re.sub(r'[^\w\s가-힣?.!]', '', text)
    if not clean_text:
        return

    filename = os.path.join(
        tempfile.gettempdir(),
        f"cooking_agent_voice_{os.getpid()}_{time.time_ns()}.mp3",
    )
    mixer_initialized = False
    try:
        with tts_lock:
            tts = gTTS(text=clean_text, lang='ko')
            tts.save(filename)
            pygame.mixer.init()
            mixer_initialized = True
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"⚠️ [TTS] 재생 실패: {e}")
    finally:
        if mixer_initialized:
            pygame.mixer.quit()
        if os.path.exists(filename):
            os.remove(filename)

# ==========================================
# 🌐 API 엔드포인트
# ==========================================
class VisionData(BaseModel):
    ingredients: list[str] = Field(default_factory=list)
    action: str = "update"


class STTData(BaseModel):
    user_text: str


def _clean_ingredients(ingredients: list[str]) -> list[str]:
    cleaned = []
    seen = set()
    for ingredient in ingredients:
        item = str(ingredient).strip()
        if not item or item in seen:
            continue
        cleaned.append(item)
        seen.add(item)
    return cleaned


def _cache_recipe_context(recipes: list[dict]) -> str:
    context_lines = []
    for i, recipe in enumerate(recipes):
        context_lines.append(f"추천요리 {i+1}: {recipe['title']}")
        context_lines.append(f"  - 전체 재료: {recipe['ingredients']}")
        context_lines.append(f"  - 요리 방법: {recipe['steps']}")
    return "\n".join(context_lines)


def _search_recipes_fallback(user_ingredients: list[str], top_k: int = 2) -> list[dict]:
    normalized_user = {
        normalize_ingredient(ingredient)
        for ingredient in user_ingredients
        if str(ingredient).strip()
    }
    if not normalized_user:
        return []

    ranked = []
    for doc in recipe_documents:
        raw_ingredients = doc.metadata.get("normalized_ingredients", "")
        recipe_ingredients = {
            item.strip()
            for item in raw_ingredients.split(",")
            if item.strip()
        }
        if not recipe_ingredients:
            continue

        overlap = recipe_ingredients & normalized_user
        if not overlap:
            continue

        missing = recipe_ingredients - normalized_user
        coverage = len(overlap) / len(recipe_ingredients)
        ranked.append((len(missing), -coverage, doc))

    ranked.sort(key=lambda item: (item[0], item[1], item[2].metadata.get("title", "")))
    recipes = []
    for _, _, doc in ranked[:top_k]:
        recipes.append({
            "title": doc.metadata["title"],
            "ingredients": doc.metadata["ingredients"],
            "steps": doc.metadata["steps"],
            "source_type": doc.metadata["source_type"],
            "similarity": 0,
        })
    return recipes


def _llm_wants_youtube_video(user_text: str) -> tuple[bool, str]:
    if is_cooking_video_query(user_text):
        return True, "rule"

    if pipe is None:
        return False, "none"

    classifier_prompt = (
        "<|im_start|>system\n"
        "너는 사용자의 요청이 유튜브 요리 영상 추천을 필요로 하는지 판단한다. "
        "사용자가 영상, 유튜브, 시연, 화면으로 보기, 조리법을 실제로 보고 싶다는 의도를 보이면 YES만 답해. "
        "일반적인 요리 질문이나 텍스트 설명만 원하는 질문이면 NO만 답해."
        "<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    try:
        outputs = pipe(
            classifier_prompt,
            max_new_tokens=8,
            do_sample=False,
            eos_token_id=pipe.tokenizer.eos_token_id,
            pad_token_id=pipe.tokenizer.pad_token_id,
        )
    except Exception as exc:
        print(f"⚠️ [YouTube] 영상 의도 판단 실패: {exc}")
        return False, "error"

    generated = outputs[0]["generated_text"]
    decision = generated.split("<|im_start|>assistant\n")[-1].strip().upper()
    return decision.startswith("YES"), "llm"


def _handle_confirmed_ingredients(
    ingredients: list[str],
    background_tasks: BackgroundTasks,
) -> dict:
    global current_ingredients, cached_rag_context, chat_history

    current_ingredients = _clean_ingredients(ingredients)
    print(f"👁️ [Vision]: {current_ingredients}")

    if not current_ingredients:
        cached_rag_context = "없음"
        return {
            "status": "success",
            "recipes_found": 0,
            "rag_loaded": vectorstore is not None or bool(recipe_documents),
            "rag_mode": rag_mode,
            "message": "인식된 재료가 없습니다.",
        }

    if not vectorstore and not recipe_documents:
        cached_rag_context = "없음"
        return {
            "status": "success",
            "recipes_found": 0,
            "rag_loaded": False,
            "message": "RAG가 아직 준비되지 않았습니다.",
        }

    print(f"🔍 RAG 검색 (재료 기반): {current_ingredients}")
    if vectorstore:
        recipes = search_recipes(vectorstore, current_ingredients, top_k=2)
    else:
        recipes = _search_recipes_fallback(current_ingredients, top_k=2)

    if recipes:
        cached_rag_context = _cache_recipe_context(recipes)
        print(f"  -> {len(recipes)}개 레시피를 컨텍스트로 캐싱함")

        first_recipe_title = recipes[0]["title"]
        opening_line = f"냉장고에 있는 재료로 만들 수 있는 {first_recipe_title} 어떠세요? 레시피가 궁금하시면 알려주세요."
    else:
        cached_rag_context = "추천할 레시피가 없습니다."
        opening_line = "가지고 계신 재료로는 추천해드릴만한 요리가 없네요. 다른 재료가 있나요?"

    print(f"🗣️ [A.X Chef]: {opening_line}")
    background_tasks.add_task(play_tts, opening_line)
    chat_history.append({"role": "assistant", "content": opening_line})

    return {
        "status": "success",
        "recipes_found": len(recipes),
        "rag_loaded": True,
        "rag_mode": rag_mode,
        "message": opening_line,
    }

@app.post("/vision")
async def update_vision(data: VisionData, background_tasks: BackgroundTasks):
    global current_ingredients, pending_ingredients, cached_rag_context

    action = (data.action or "update").strip().lower()
    ingredients = _clean_ingredients(data.ingredients)

    if action == "ask_confirmation":
        pending_ingredients = ingredients
        if not pending_ingredients:
            return {"status": "ignored", "reason": "no_ingredients"}

        ingredient_text = ", ".join(pending_ingredients)
        confirmation_line = f"{ingredient_text} 재료가 맞나요? 맞으면 엄지척, 아니면 주먹을 보여주세요."
        print(f"🗣️ [A.X Chef]: {confirmation_line}")
        background_tasks.add_task(play_tts, confirmation_line)
        return {
            "status": "waiting_confirmation",
            "ingredients": pending_ingredients,
            "message": confirmation_line,
        }

    if action == "confirm":
        confirmed = ingredients or pending_ingredients
        pending_ingredients = []
        return _handle_confirmed_ingredients(confirmed, background_tasks)

    if action == "reject":
        current_ingredients = []
        pending_ingredients = []
        cached_rag_context = "없음"
        rejection_line = "알겠습니다. 재료를 다시 인식해볼게요."
        print(f"🗣️ [A.X Chef]: {rejection_line}")
        background_tasks.add_task(play_tts, rejection_line)
        return {"status": "rejected", "message": rejection_line}

    pending_ingredients = []
    return _handle_confirmed_ingredients(ingredients, background_tasks)


@app.get("/youtube-preview")
async def youtube_preview(query: str = ""):
    text = query.strip()
    youtube_api_key = get_runtime_env("YOUTUBE_API_KEY")
    wants_youtube = is_cooking_video_query(text) if text else False
    youtube_status = {
        "requested": wants_youtube,
        "intent_source": "rule" if wants_youtube else "none",
        "enabled": bool(youtube_api_key),
        "message": "",
    }

    if not text or not wants_youtube:
        return {
            "requested": wants_youtube,
            "video_recommendation": None,
            "youtube_status": youtube_status,
        }

    if not youtube_api_key:
        youtube_status["message"] = "YouTube API 키가 설정되지 않아 영상을 가져오지 못했습니다."
        return {
            "requested": True,
            "video_recommendation": None,
            "youtube_status": youtube_status,
        }

    try:
        video_recommendation = await run_in_threadpool(
            find_best_youtube_segment,
            text,
            youtube_api_key,
        )
        if video_recommendation:
            youtube_status["message"] = "관련 유튜브 영상을 찾았습니다."
        else:
            youtube_error = get_last_youtube_error()
            youtube_status["message"] = (
                f"YouTube API 호출 실패: {youtube_error}"
                if youtube_error
                else "유튜브에서 관련 영상을 찾지 못했습니다."
            )
        return {
            "requested": True,
            "video_recommendation": video_recommendation,
            "youtube_status": youtube_status,
        }
    except Exception as e:
        print(f"⚠️ [YouTube] 프리뷰 생성 실패: {e}")
        youtube_status["message"] = f"YouTube 프리뷰 생성 실패: {e}"
        return {
            "requested": True,
            "video_recommendation": None,
            "youtube_status": youtube_status,
        }


@app.post("/cancel")
async def cancel_generation():
    was_active = is_generation_active()
    generation_cancel_event.set()
    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
    except Exception as exc:
        print(f"⚠️ [Cancel] TTS 중단 실패: {exc}")
    return {
        "status": "cancelling" if was_active else "idle",
        "was_active": was_active,
    }


@app.post("/ask")
async def ask_chef(data: STTData, background_tasks: BackgroundTasks):
    global chat_history, cached_rag_context
    if pipe is None:
        raise HTTPException(status_code=503, detail="LLM model is not loaded yet.")

    youtube_api_key = get_runtime_env("YOUTUBE_API_KEY")
    wants_youtube, youtube_intent_source = _llm_wants_youtube_video(data.user_text)
    
    # 💡 저장된 RAG 검색 결과를 가져와 사용합니다.
    rag_context = cached_rag_context

    ing_str = ", ".join(current_ingredients) if current_ingredients else "없음"
    
    # 프롬프트 구성
    prompt_template = SYSTEM_PROMPT.format(rag_context=rag_context)
    youtube_instruction = ""
    if wants_youtube:
        youtube_instruction = (
            "\n사용자가 유튜브 영상 또는 시연 영상을 요청했습니다. "
            "시스템이 별도로 유튜브 영상을 검색해서 화면에 붙일 예정이니, "
            "절대 '영상은 제공할 수 없습니다'라고 말하지 마세요. "
            "사람 셰프처럼 자연스럽게 지금 필요한 조리 포인트 한 단계만 설명하고 '아래 영상도 같이 확인해보세요'라고 말하세요."
        )

    prompt = f"<|im_start|>system\n{prompt_template}\n현재 인식된 재료: {ing_str}{youtube_instruction}<|im_end|>\n"
    
    # 이전 대화 추가
    for hist in chat_history[-4:]:
        prompt += f"<|im_start|>{hist['role']}\n{hist['content']}<|im_end|>\n"
    
    # 현재 질문 추가
    prompt += f"<|im_start|>user\n{data.user_text}<|im_end|>\n<|im_start|>assistant\n"
    
    try:
        llm_answer = await run_in_threadpool(generate_llm_answer, prompt)
    except GenerationCancelled:
        return {
            "answer": "응답 생성을 중단했습니다.",
            "cancelled": True,
            "video_recommendation": None,
            "youtube_status": {
                "requested": wants_youtube,
                "intent_source": youtube_intent_source,
                "enabled": bool(youtube_api_key),
                "message": "",
            },
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if wants_youtube and any(
        phrase in llm_answer
        for phrase in (
            "영상은 제공할 수 없습니다",
            "영상을 제공할 수 없습니다",
            "동영상은 제공할 수 없습니다",
            "동영상을 제공할 수 없습니다",
        )
    ):
        llm_answer = "요청하신 조리 영상도 같이 찾아볼게요. 아래 영상이 뜨면 같이 확인해보세요. 다 하셨으면 말씀해 주세요."
    
    # 대화 기록 업데이트
    chat_history.append({"role": "user", "content": data.user_text})
    chat_history.append({"role": "assistant", "content": llm_answer})

    print(f"🔥 [A.X Chef]: {llm_answer}")
    background_tasks.add_task(play_tts, llm_answer)

    youtube_status = {
        "requested": wants_youtube,
        "intent_source": youtube_intent_source,
        "enabled": bool(youtube_api_key),
        "message": "",
    }
    video_recommendation = None

    if wants_youtube and not youtube_api_key:
        youtube_status["message"] = "YouTube API 키가 설정되지 않아 영상을 가져오지 못했습니다."

    if wants_youtube and youtube_api_key:
        try:
            video_recommendation = await run_in_threadpool(
                find_best_youtube_segment,
                data.user_text,
                youtube_api_key,
            )
            if video_recommendation:
                youtube_status["message"] = "관련 유튜브 영상을 찾았습니다."
            else:
                youtube_error = get_last_youtube_error()
                youtube_status["message"] = (
                    f"YouTube API 호출 실패: {youtube_error}"
                    if youtube_error
                    else "유튜브에서 관련 영상을 찾지 못했습니다."
                )
        except Exception as e:
            print(f"⚠️ [YouTube] 추천 생성 실패: {e}")
            youtube_status["message"] = f"YouTube 추천 생성 실패: {e}"

    return {
        "answer": llm_answer,
        "video_recommendation": video_recommendation,
        "youtube_status": youtube_status,
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("LLM_HOST", "127.0.0.1")
    port = int(os.getenv("LLM_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
