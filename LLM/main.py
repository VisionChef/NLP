from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException
from pydantic import BaseModel
import torch
from transformers import pipeline
from huggingface_hub import login
import os
import re
from gtts import gTTS
import pygame
import time
import shutil
from contextlib import asynccontextmanager

from Rag.rag import (
    load_recipes, 
    build_vectorstore, 
    search_recipes, 
    BOOK_RECIPES_FILE, 
    TRENDING_RECIPES_FILE, 
    CHROMA_DIR
)

# ⚠️ 환경 설정
os.environ["HF_HOME"] = os.path.abspath("hf_cache")
os.environ["HF_HUB_DISABLE_EXPERIMENTAL_XET"] = "1"

# 🔑 허깅페이스 토큰 (환경 변수에서 로드)
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Hugging Face 토큰이 설정되지 않았습니다. HF_TOKEN 환경 변수를 설정해주세요.")
login(token=HF_TOKEN)

# ==========================================
# ⚙️ 설정, 모델 로드, RAG 초기화
# ==========================================
pipe = None
vectorstore = None # RAG 벡터 저장소
LLM_MODEL_ID = "skt/A.X-4.0-Light"
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 실행
    global pipe, vectorstore
    print("🚀 서버 시작...")
    
    # LLM 모델 로드
    print(f"🧠 {LLM_MODEL_ID} 로딩 중...")
    pipe = pipeline(
        "text-generation", 
        model=LLM_MODEL_ID, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    print("✅ A.X 모델 준비 완료!")

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
cached_rag_context = "없음"
chat_history = [] 

# ==========================================
# 🔊 TTS 재생 함수
# ==========================================
def play_tts(text: str):
    clean_text = re.sub(r'[^\w\s가-힣?.!]', '', text)
    filename = f"voice_{int(time.time())}.mp3"
    try:
        tts = gTTS(text=clean_text, lang='ko')
        tts.save(filename)
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    finally:
        pygame.mixer.quit()
        if os.path.exists(filename):
            os.remove(filename)

# ==========================================
# 🌐 API 엔드포인트
# ==========================================
class VisionData(BaseModel): ingredients: list[str]
class STTData(BaseModel): user_text: str

@app.post("/vision")
async def update_vision(data: VisionData, background_tasks: BackgroundTasks):
    global current_ingredients, cached_rag_context, vectorstore, chat_history
    current_ingredients = data.ingredients
    print(f"👁️ [Vision]: {current_ingredients}")
    
    # 💡 재료가 업데이트될 때 RAG 검색을 수행하고, 결과에 따라 대화를 시작합니다.
    if vectorstore and current_ingredients:
        print(f"🔍 RAG 검색 (재료 기반): {current_ingredients}")
        recipes = search_recipes(vectorstore, current_ingredients, top_k=2)
        
        if recipes:
            # RAG 컨텍스트 캐싱
            context_lines = []
            for i, recipe in enumerate(recipes):
                context_lines.append(f"추천요리 {i+1}: {recipe['title']}")
                context_lines.append(f"  - 전체 재료: {recipe['ingredients']}")
                context_lines.append(f"  - 요리 방법: {recipe['steps']}")
            cached_rag_context = "\n".join(context_lines)
            print(f"  -> {len(recipes)}개 레시피를 컨텍스트로 캐싱함")

            # 🚀 대화 시작 메시지 생성 및 재생
            first_recipe_title = recipes[0]['title']
            opening_line = f"냉장고에 있는 재료로 만들 수 있는 {first_recipe_title} 어떠세요? 레시피가 궁금하시면 알려주세요."
            
            print(f"🗣️ [A.X Chef]: {opening_line}")
            background_tasks.add_task(play_tts, opening_line)

            # 대화 기록에 어시스턴트의 첫 메시지 추가
            chat_history.append({"role": "assistant", "content": opening_line})

        else:
            cached_rag_context = "추천할 레시피가 없습니다."
            # 추천할 것이 없을 때도 사용자에게 알려줄 수 있습니다.
            opening_line = "가지고 계신 재료로는 추천해드릴만한 요리가 없네요. 다른 재료가 있나요?"
            print(f"🗣️ [A.X Chef]: {opening_line}")
            background_tasks.add_task(play_tts, opening_line)
            chat_history.append({"role": "assistant", "content": opening_line})
    else:
        cached_rag_context = "없음"

    return {"status": "success"}

@app.post("/ask")
async def ask_chef(data: STTData, background_tasks: BackgroundTasks):
    global chat_history, cached_rag_context
    
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
    
    return {"answer": llm_answer}