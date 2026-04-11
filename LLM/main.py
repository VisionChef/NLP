from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import torch
from transformers import pipeline
from huggingface_hub import login
import os
import re
from gtts import gTTS
import pygame
import time

# ⚠️ 환경 설정
os.environ["HF_HOME"] = os.path.abspath("hf_cache")
os.environ["HF_HUB_DISABLE_EXPERIMENTAL_XET"] = "1"

# 🔑 허깅페이스 토큰 (환경 변수에서 로드)
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Hugging Face 토큰이 설정되지 않았습니다. HF_TOKEN 환경 변수를 설정해주세요.")
login(token=HF_TOKEN)

app = FastAPI()

# ==========================================
# ⚙️ 설정 및 SKT A.X 모델 로드
# ==========================================
LLM_MODEL_ID = "skt/A.X-4.0-Light"
SYSTEM_PROMPT = """너는 사용자와 소통하며 요리하는 전문 셰프야. 아래 규칙을 반드시 지켜줘.
1. 절대 *, #, - 같은 기호를 사용하지 말고 구어체로만 말해.
2. 한 번에 모든 단계를 말하지 말고 딱 한 단계씩만 끊어서 설명해.
3. 단계가 끝나면 "다 하셨으면 말씀해 주세요"라고 물어봐서 대화를 유도해.
"""

current_ingredients = []
chat_history = [] 

print(f"🧠 {LLM_MODEL_ID} 로딩 중...")
pipe = pipeline(
    "text-generation", 
    model=LLM_MODEL_ID, 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)
print("✅ A.X 서버 준비 완료!")

# ==========================================
# 🔊 TTS 재생 함수
# ==========================================
def play_tts(text: str):
    # 말하기 전 특수 기호 제거
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
        pygame.mixer.quit() 
        if os.path.exists(filename): os.remove(filename)
    except Exception as e:
        print(f"⚠️ [TTS 에러]: {e}")

# ==========================================
# 🌐 API 엔드포인트
# ==========================================
class VisionData(BaseModel): ingredients: list[str]
class STTData(BaseModel): user_text: str

@app.post("/vision")
async def update_vision(data: VisionData):
    global current_ingredients
    current_ingredients = data.ingredients
    print(f"👁️ [Vision]: {current_ingredients}")
    return {"status": "success"}

@app.post("/ask")
async def ask_chef(data: STTData, background_tasks: BackgroundTasks):
    global chat_history
    ing_str = ", ".join(current_ingredients) if current_ingredients else "없음"
    
    # 1. A.X-4.0-Light 전용 ChatML 프롬프트 구성
    # <|im_start|>system...<|im_end|> 구조를 따릅니다.
    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}\n현재 인식된 재료: {ing_str}<|im_end|>\n"
    
    # 이전 대화 추가
    for i, hist in enumerate(chat_history[-4:]):
        role = "user" if i % 2 == 0 else "assistant"
        prompt += f"<|im_start|>{role}\n{hist}<|im_end|>\n"
    
    # 현재 질문 추가
    prompt += f"<|im_start|>user\n{data.user_text}<|im_end|>\n<|im_start|>assistant\n"
    
    # 2. 답변 생성
    outputs = pipe(
        prompt, 
        max_new_tokens=256, 
        do_sample=True, 
        temperature=0.7,
        eos_token_id=27, # 올려주신 JSON의 <|im_end|> ID
        pad_token_id=1
    )
    
    # 3. 답변 파싱
    full_text = outputs[0]["generated_text"]
    llm_answer = full_text.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
    
    # 대화 기록 업데이트
    chat_history.append(data.user_text)
    chat_history.append(llm_answer)

    print(f"🔥 [A.X Chef]: {llm_answer}")
    background_tasks.add_task(play_tts, llm_answer)
    
    return {"answer": llm_answer}