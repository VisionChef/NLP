import os
import re
from getpass import getpass
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional
import tempfile
import threading

# ⚠️ Hugging Face / Transformers 캐시는 관련 라이브러리 import 전에 잡아둔다.
DEFAULT_LOCAL_MODEL_DIR = r"D:\models\skt_A.X-4.0-Light"
DEFAULT_HF_HOME = r"D:\models\hf_cache"
os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
os.environ.setdefault("HF_HUB_DISABLE_EXPERIMENTAL_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login, snapshot_download
from gtts import gTTS
import pygame
import time
from starlette.concurrency import run_in_threadpool

from Rag.rag import (
    load_recipes, 
    build_vectorstore, 
    search_recipes, 
    BOOK_RECIPES_FILE, 
    TRENDING_RECIPES_FILE, 
    CHROMA_DIR
)
from youtube_api import find_best_youtube_segment, is_cooking_video_query

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# ==========================================
# ⚙️ 설정, 모델 로드, RAG 초기화
# ==========================================
pipe = None
vectorstore = None # RAG 벡터 저장소
loaded_model_source = None
loaded_quantization = "none"
tts_lock = threading.Lock()
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "skt/A.X-4.0-Light")
LLM_LOCAL_MODEL_DIR = os.getenv("LLM_LOCAL_MODEL_DIR", DEFAULT_LOCAL_MODEL_DIR)
LLM_LOAD_IN_8BIT = os.getenv("LLM_LOAD_IN_8BIT", "0").strip().lower() in {"1", "true", "yes", "on"}
SYSTEM_PROMPT = """너는 사용자와 소통하며 요리하는 전문 셰프이자, 주어진 문서에 대해 답변하는 전문가야.
- 만약 사용자의 질문이 요리와 관련 있다면, 전문 셰프처럼 아래 규칙을 지켜서 답변해줘.
  1. 절대 *, #, - 같은 기호를 사용하지 말고 구어체로만 말해.
  2. 한 번에 모든 단계를 말하지 말고 딱 한 단계씩만 끊어서 설명해.
  3. 단계가 끝나면 "다 하셨으면 말씀해 주세요"라고 물어봐서 대화를 유도해.
- 만약 사용자의 질문이 문서 내용과 관련 있다면, 아래 "참고 문서 내용"을 바탕으로 친절하게 설명해줘.
---
[참고 문서 내용]
{rag_context}
---
"""


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
    global pipe, vectorstore, loaded_model_source
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
    all_docs = book_docs + trending_docs
    vectorstore = build_vectorstore(all_docs, chroma_path)

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
        "rag_loaded": vectorstore is not None,
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


def _handle_confirmed_ingredients(
    ingredients: list[str],
    background_tasks: BackgroundTasks,
) -> dict:
    global current_ingredients, cached_rag_context, chat_history

    current_ingredients = _clean_ingredients(ingredients)
    print(f"👁️ [Vision]: {current_ingredients}")

    if not current_ingredients:
        cached_rag_context = "없음"
        return {"status": "success", "recipes_found": 0}

    if not vectorstore:
        cached_rag_context = "없음"
        return {"status": "success", "recipes_found": 0, "rag_loaded": False}

    print(f"🔍 RAG 검색 (재료 기반): {current_ingredients}")
    recipes = search_recipes(vectorstore, current_ingredients, top_k=2)

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

    return {"status": "success", "recipes_found": len(recipes)}

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
        return {"status": "waiting_confirmation", "ingredients": pending_ingredients}

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
        return {"status": "rejected"}

    pending_ingredients = []
    return _handle_confirmed_ingredients(ingredients, background_tasks)

@app.post("/ask")
async def ask_chef(data: STTData, background_tasks: BackgroundTasks):
    global chat_history, cached_rag_context
    if pipe is None:
        raise HTTPException(status_code=503, detail="LLM model is not loaded yet.")
    
    # 💡 저장된 RAG 검색 결과를 가져와 사용합니다.
    rag_context = cached_rag_context

    ing_str = ", ".join(current_ingredients) if current_ingredients else "없음"
    
    # 프롬프트 구성
    prompt_template = SYSTEM_PROMPT.format(rag_context=rag_context)
    prompt = f"<|im_start|>system\n{prompt_template}\n현재 인식된 재료: {ing_str}<|im_end|>\n"
    
    # 이전 대화 추가
    for hist in chat_history[-4:]:
        prompt += f"<|im_start|>{hist['role']}\n{hist['content']}<|im_end|>\n"
    
    # 현재 질문 추가
    prompt += f"<|im_start|>user\n{data.user_text}<|im_end|>\n<|im_start|>assistant\n"
    
    # 답변 생성
    outputs = pipe(
        prompt, 
        max_new_tokens=512,
        do_sample=True, 
        temperature=0.7,
        eos_token_id=pipe.tokenizer.eos_token_id,
        pad_token_id=pipe.tokenizer.pad_token_id
    )
    
    # 답변 파싱
    full_text = outputs[0]["generated_text"]
    llm_answer = full_text.split("<|im_start|>assistant\n")[-1].strip()
    
    # 대화 기록 업데이트
    chat_history.append({"role": "user", "content": data.user_text})
    chat_history.append({"role": "assistant", "content": llm_answer})

    print(f"🔥 [A.X Chef]: {llm_answer}")
    background_tasks.add_task(play_tts, llm_answer)

    video_recommendation = None
    if YOUTUBE_API_KEY and is_cooking_video_query(data.user_text):
        try:
            video_recommendation = await run_in_threadpool(
                find_best_youtube_segment,
                data.user_text,
                YOUTUBE_API_KEY,
            )
        except Exception as e:
            print(f"⚠️ [YouTube] 추천 생성 실패: {e}")

    return {"answer": llm_answer, "video_recommendation": video_recommendation}


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("LLM_HOST", "127.0.0.1")
    port = int(os.getenv("LLM_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
