"""
레시피 RAG 파이프라인 (검색 결과만 반환)
==============================
필요한 패키지:
    pip install langchain langchain-chroma langchain-huggingface chromadb sentence-transformers python-dotenv
"""

import json
import os
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_ENABLE_TQDM_MULTIPROCESSING"] = "0"
os.environ["HF_HOME"] = os.environ.get("HF_HOME", r"C:\hf_cache_clean")
if os.getenv("RAG_ALLOW_DOWNLOAD", "0").strip().lower() not in {"1", "true", "yes", "on"}:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ─────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────
MODULE_DIR = Path(__file__).resolve().parent
BOOK_RECIPES_FILE     = "recipes.json"
TRENDING_RECIPES_FILE = "trending_recipes.json"
CHROMA_DIR            = "chroma_db"

MIN_SIMILARITY_SCORE = 0.85


# ─────────────────────────────────────
# 재료 정규화 (사용자 입력에도 동일하게 적용)
# ─────────────────────────────────────
SYNONYM_MAP = {
    "계란":             "달걀",
    "달걀":             "달걀",
    "꽃소금":           "소금",
    "맛소금":           "소금",
    "천일염":           "소금",
    "소금":             "소금",
    "진간장":           "진간장",
    "국간장":           "국간장",
    "양조간장":         "양조간장",
    "굴소스":           "굴소스",
    "쯔유":             "쯔유",
    "굵은고춧가루":     "고춧가루",
    "굵은 고춧가루":    "고춧가루",
    "고운 고춧가루":    "고춧가루",
    "고춧가루":         "고춧가루",
    "풋고추":           "고추",
    "홍고추":           "고추",
    "청양고추":         "고추",
    "고추":             "고추",
    "후추가루":         "후춧가루",
    "후추 가루":        "후춧가루",
    "후춧가루":         "후춧가루",
    "후추":             "후춧가루",
    "흑후추":           "후춧가루",
    "간마늘":           "마늘",
    "간 마늘":          "마늘",
    "다진마늘":         "마늘",
    "다진 마늘":        "마늘",
    "마늘":             "마늘",
    "대파":             "파",
    "쪽파":             "파",
    "실파":             "파",
    "돼지고기":         "돼지고기",
    "삼겹살":           "돼지고기",
    "목살":             "돼지고기",
    "돼지목살":         "돼지고기",
    "돼지 목살":        "돼지고기",
    "돼지삼겹살":       "돼지고기",
    "돼지 삼겹살":      "돼지고기",
    "간돼지고기":       "돼지고기",
    "간 돼지고기":      "돼지고기",
    "다진돼지고기":     "돼지고기",
    "다진 돼지고기":    "돼지고기",
    "우삼겹":           "돼지고기",
    "소고기":           "소고기",
    "간소고기":         "소고기",
    "간 소고기":        "소고기",
    "다진소고기":       "소고기",
    "다진 소고기":      "소고기",
    "소고기다짐육":     "소고기",
    "소고기 다짐육":    "소고기",
    "양지":             "소고기",
    "사태":             "소고기",
    "양지머리":         "소고기",
    "닭고기":           "닭고기",
    "영계":             "닭고기",
    "닭가슴살":         "닭고기",
    "닭 가슴살":        "닭고기",
    "닭다리살":         "닭고기",
    "닭 다리살":        "닭고기",
    "신김치":           "김치",
    "신 김치":          "김치",
    "묵은지":           "김치",
    "배추김치":         "김치",
    "김치":             "김치",
    "단호박":           "호박",
    "늙은호박":         "호박",
    "늙은 호박":        "호박",
    "애호박":           "호박",
    "주키니호박":       "호박",
    "주키니 호박":      "호박",
    "호박":             "호박",
    "표고버섯":         "버섯",
    "느타리버섯":       "버섯",
    "팽이버섯":         "버섯",
    "버섯":             "버섯",
    "쌀":               "쌀",
    "불린쌀":           "쌀",
    "불린 쌀":          "쌀",
    "찹쌀":             "찹쌀",
    "불린찹쌀":         "찹쌀",
    "불린 찹쌀":        "찹쌀",
    "찹쌀가루":         "찹쌀가루",
    "가래떡":           "떡",
    "흰가래떡":         "떡",
    "떡볶이용떡":       "떡",
    "떡볶이용 떡":      "떡",
    "떡":               "떡",
    "수삼":             "인삼",
    "건삼":             "인삼",
    "홍삼":             "인삼",
    "인삼":             "인삼",
    "얼갈이배추":       "배추",
    "얼갈이 배추":      "배추",
    "배추":             "배추",
    "불린고사리":       "고사리",
    "불린 고사리":      "고사리",
    "고사리":           "고사리",
    "참기름":           "참기름",
    "들기름":           "들기름",
    "식용유":           "식용유",
    "김밥김":           "생김",
    "소세지":           "햄",
    "소시지":           "햄",
}


def clean_inline_newlines(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_ingredient(text: str) -> str:
    text = text.strip()
    text = clean_inline_newlines(text)
    text = re.sub(r"\([^)]*\)", "", text)
    text = text.replace("·", " ").replace(":", " ")
    text = re.sub(r"[,\[\]{}]+", " ", text)

    unit_pattern = (
        r"\d+(?:[./]\d+)?\s*"
        r"(g|kg|ml|l|컵|큰술|작은술|술|개|장|줄|공기|인분|쪽|톨|봉|줌|약간|적당량)"
    )
    text = re.sub(unit_pattern, "", text)
    text = re.sub(r"\d+", "", text)

    for word in ["약간", "적당량", "약간의", "조금", "조금의"]:
        text = text.replace(word, "")

    text = re.sub(r"\s+", " ", text).strip()

    if text in SYNONYM_MAP:
        return SYNONYM_MAP[text]

    text_nospace = text.replace(" ", "")
    if text_nospace in SYNONYM_MAP:
        return SYNONYM_MAP[text_nospace]

    return text


# ─────────────────────────────────────
# 임베딩 모델
# ─────────────────────────────────────
def get_embeddings() -> HuggingFaceEmbeddings:
    model_name = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-m3")
    allow_download = os.getenv("RAG_ALLOW_DOWNLOAD", "0").strip().lower() in {"1", "true", "yes", "on"}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=os.getenv("HF_HOME"),
        model_kwargs={"local_files_only": not allow_download},
    )


# ─────────────────────────────────────
# 1단계: JSON 로드 → Document 변환
# ─────────────────────────────────────
def load_recipes(path: str, source_type: str) -> list[Document]:
    with open(path, "r", encoding="utf-8") as f:
        recipes = json.load(f)

    documents = []
    for recipe in recipes:
        ingredients_text = ", ".join(sorted(recipe["normalized_ingredients"]))
        content = f"재료: {ingredients_text}"
        steps_text = "\n".join(recipe["steps"])

        metadata = {
            "id":                     recipe["id"],
            "title":                  recipe["title"],
            "ingredients":            ", ".join(recipe["ingredients"]),
            "normalized_ingredients": ingredients_text,
            "steps":                  steps_text,
            "source_type":            source_type,
        }

        documents.append(Document(page_content=content, metadata=metadata))

    print(f"✅ [{source_type}] 레시피 {len(documents)}개 로드 완료")
    return documents


# ─────────────────────────────────────
# 2단계: 벡터 DB 구축
# ─────────────────────────────────────
def build_vectorstore(documents: list[Document], persist_dir: str) -> Chroma:
    embeddings = get_embeddings()
    collection_meta = {"hnsw:space": "cosine"}

    if Path(persist_dir).exists():
        print(f"📂 기존 벡터 DB 로드: {persist_dir}")
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_metadata=collection_meta,
        )
    else:
        print("🔨 벡터 DB 새로 구축 중...")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_metadata=collection_meta,
        )
        print(f"✅ 벡터 DB 저장 완료: {persist_dir}")

    return vectorstore


# ─────────────────────────────────────
# 3단계: 재료 기반 레시피 검색 + 필터링
# ─────────────────────────────────────
def search_recipes(
    vectorstore: Chroma,
    user_ingredients: list[str],
    top_k: int = 5,
    min_score: float = MIN_SIMILARITY_SCORE,
) -> list[dict]:
    # 사용자 입력 재료 정규화
    normalized_user = set(normalize_ingredient(i) for i in user_ingredients)

    query = f"재료: {', '.join(sorted(normalized_user))}"
    results = vectorstore.similarity_search_with_score(query, k=top_k)

    recipes_found = []
    for doc, score in results:
        similarity = round(1 - score, 3)

        # 유사도 필터
        if similarity < min_score:
            print(f"유사도 미달 제외 ({similarity} < {min_score}): {doc.metadata['title']}")
            continue

        # 재료 완전 보유 여부 체크
        recipe_ingredients = set(
            i.strip()
            for i in doc.metadata["normalized_ingredients"].split(",")
            if i.strip()
        )
        missing = recipe_ingredients - normalized_user
        if missing:
            print(f"재료 부족 제외 (없는 재료: {missing}): {doc.metadata['title']}")
            continue

        recipes_found.append({
            "title":       doc.metadata["title"],
            "ingredients": doc.metadata["ingredients"],
            "steps":       doc.metadata["steps"],
            "source_type": doc.metadata["source_type"],
            "similarity":  similarity,
        })

    return recipes_found


# ─────────────────────────────────────
# 결과 포맷 (SKT A.X로 넘길 형태)
# ─────────────────────────────────────
def format_results(
    user_ingredients: list[str],
    recipes: list[dict],
) -> dict:
    return {
        "user_ingredients": user_ingredients,
        "recommended_recipes": [
            {
                "title":       r["title"],
                "ingredients": r["ingredients"],
                "steps":       r["steps"],
                "source_type": r["source_type"],
                "similarity":  r["similarity"],
            }
            for r in recipes
        ],
    }


# ─────────────────────────────────────
# 실행
# ─────────────────────────────────────
if __name__ == "__main__":
    # 1. 두 JSON 로드
    book_docs = load_recipes(str(MODULE_DIR / BOOK_RECIPES_FILE), source_type="Baek_Book")
    trending_docs = load_recipes(str(MODULE_DIR / TRENDING_RECIPES_FILE), source_type="trending")
    all_docs = book_docs + trending_docs

    # 2. 벡터 DB 구축
    vectorstore = build_vectorstore(all_docs, str(MODULE_DIR / CHROMA_DIR))

    # 3. 재료 입력
    print("\n냉장고에 있는 재료를 입력하세요 (띄어쓰기 또는 쉼표로 구분):")
    user_input = input("재료: ").strip()
    detected = [item.strip() for item in user_input.replace(",", " ").split()]

    # 4. RAG 검색 + 필터링
    results = search_recipes(vectorstore, detected, top_k=5, min_score=MIN_SIMILARITY_SCORE)

    # 5. 결과 출력
    output = format_results(detected, results)

    if not results:
        print("\n⚠️  만들 수 있는 레시피가 없어요.")
    else:
        print("\n검색된 레시피:")
        for r in output["recommended_recipes"]:
            print(f"  - [{r['source_type']}] {r['title']} (유사도: {r['similarity']})")

    print("\n[SKT A.X로 넘길 결과]")
    print(json.dumps(output, ensure_ascii=False, indent=2))
