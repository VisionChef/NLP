import json
import re
import sys
from pathlib import Path
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from synonym_map import SYNONYM_MAP

RANGES = [
    (0, 28),
    (57, 91),
    (94, 126),
    (129, 156),
    (162, 190),
    (252, 283),
    (286, 317),
    (383, 412)

    
# r"C:\VisionChef\upstage_outputs\***" 경로 설정 확인하고 코드 실행하기!!!
]

BASE_DIR = Path(r"C:\VisionChef\upstage_outputs\4-4")
HTML_PATH = BASE_DIR / "content.html"
OUTPUT_PATH = BASE_DIR / "recipes.json"


def html_to_text(tag) -> str:
    for br in tag.find_all("br"):
        br.replace_with("\n")
    text = tag.get_text("\n", strip=True)
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text


def clean_inline_newlines(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_step_text(text: str) -> str:
    text = clean_inline_newlines(text)
    text = re.sub(r"^\d+\.\s*", "", text)
    return text.strip()


def is_footer_or_figure(tag) -> bool:
    return tag.name in {"figure", "footer"}


def is_tip_text(text: str) -> bool:
    bad_keywords = ["백종원의", "Tip", "팁"]
    return any(k in text for k in bad_keywords)


def is_recipe_header(text: str) -> bool:
    return "재료" in text


def looks_like_title(text: str) -> bool:
    if not text:
        return False
    if "재료" in text or "Tip" in text or "백종원의" in text:
        return False
    if len(text) > 30:
        return False
    if "\n" in text:
        return False
    return True


def looks_like_step(text: str) -> bool:
    if not text:
        return False
    if is_tip_text(text):
        return False
    if re.fullmatch(r"\d+", text):
        return False

    endings = [
        "다.", "한다.", "낸다.", "넣는다.", "볶는다.",
        "썬다.", "올린다.", "말아준다.", "식힌다.",
        "부친다.", "끓인다.", "섞는다.", "비빈다."
    ]
    return any(text.endswith(e) or e in text for e in endings)


def extract_ingredients_from_text(text: str) -> list[str]:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    cleaned = []
    for line in lines:
        if "재료" in line:
            continue
        cleaned.append(line)
    return cleaned


def extract_table_rows(table_tag) -> list[str]:
    rows = []
    for tr in table_tag.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        if cells:
            rows.append(" ".join(cells))
    return rows


def normalize_ingredient(text: str) -> str:
    text = text.strip()
    text = clean_inline_newlines(text)

    # 괄호 안 내용 제거
    text = re.sub(r"\([^)]*\)", "", text)

    text = text.replace("·", " ")
    text = text.replace(":", " ")
    text = re.sub(r"[,\[\]{}]+", " ", text)

    # 수량/단위 제거
    unit_pattern = (
        r"\d+(?:[./]\d+)?\s*"
        r"(g|kg|ml|l|컵|큰술|작은술|술|개|장|줄|공기|인분|쪽|톨|봉|줌|약간|적당량)"
    )
    text = re.sub(unit_pattern, "", text)
    text = re.sub(r"\d+", "", text)

    remove_words = ["약간", "적당량", "약간의", "조금", "조금의"]
    for word in remove_words:
        text = text.replace(word, "")

    text = re.sub(r"\s+", " ", text).strip()

    # ① 공백 있는 상태로 먼저 시도
    if text in SYNONYM_MAP:
        return SYNONYM_MAP[text]

    # ② 공백 제거 후 재시도 ('신 김치' → '신김치' 처리)
    text_nospace = text.replace(" ", "")
    if text_nospace in SYNONYM_MAP:
        return SYNONYM_MAP[text_nospace]

    # ③ 부분 포함 기반 보정 (공백 있는 버전)
    for key, value in SYNONYM_MAP.items():
        if key in text:
            return value

    # ④ 부분 포함 기반 보정 (공백 제거 버전)
    for key, value in SYNONYM_MAP.items():
        if key.replace(" ", "") in text_nospace:
            return value

    return text


def build_normalized_ingredients(ingredients: list[str]) -> list[str]:
    normalized = []
    seen = set()

    for item in ingredients:
        norm = normalize_ingredient(item)
        if norm and norm not in seen:
            normalized.append(norm)
            seen.add(norm)

    return normalized


def parse_recipe_block(tags, recipe_idx: int):
    texts = []

    for tag in tags:
        if not tag.get("id"):
            continue
        if is_footer_or_figure(tag):
            continue

        text = html_to_text(tag)
        if text:
            texts.append((int(tag["id"]), tag.name, text, tag))

    ingredient_header_idx = None
    for i, (_, _, text, _) in enumerate(texts):
        if is_recipe_header(text):
            ingredient_header_idx = i
            break

    before_ingredients = texts[:ingredient_header_idx] if ingredient_header_idx is not None else texts

    title = None
    for _, _, text, _ in before_ingredients:
        if looks_like_title(text):
            title = clean_inline_newlines(text)
            break

    if title is None and before_ingredients:
        title = clean_inline_newlines(min(before_ingredients, key=lambda x: len(x[2]))[2])

    ingredients = []
    step_start_idx = None

    if ingredient_header_idx is not None:
        for i in range(ingredient_header_idx + 1, len(texts)):
            _, tag_name, text, tag = texts[i]

            if looks_like_step(text):
                step_start_idx = i
                break

            if tag_name == "table":
                ingredients.extend(extract_table_rows(tag))
            else:
                ingredients.extend(extract_ingredients_from_text(text))

    ingredients = [clean_inline_newlines(x) for x in ingredients if clean_inline_newlines(x)]
    normalized_ingredients = build_normalized_ingredients(ingredients)

    steps = []
    if step_start_idx is not None:
        for i in range(step_start_idx, len(texts)):
            _, _, text, _ = texts[i]

            if is_tip_text(text):
                continue

            if looks_like_step(text):
                cleaned = clean_step_text(text)
                if cleaned:
                    steps.append(cleaned)

    return {
        "id": f"recipe_{recipe_idx:03d}",
        "title": title or "",
        "ingredients": ingredients,
        "normalized_ingredients": normalized_ingredients,
        "steps": steps,  # 리스트로 저장
    }


def main():
    if not HTML_PATH.exists():
        raise FileNotFoundError(f"content.html 파일이 없습니다: {HTML_PATH}")

    html = HTML_PATH.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    all_tags = []
    for tag in soup.find_all(True):
        tag_id = tag.get("id")
        if tag_id is not None and str(tag_id).isdigit():
            all_tags.append(tag)

    recipes = []
    for idx, (start_id, end_id) in enumerate(RANGES, start=1):
        recipe_tags = [
            tag for tag in all_tags
            if start_id <= int(tag["id"]) <= end_id
        ]
        recipe = parse_recipe_block(recipe_tags, idx)
        recipes.append(recipe)

    OUTPUT_PATH.write_text(
        json.dumps(recipes, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()