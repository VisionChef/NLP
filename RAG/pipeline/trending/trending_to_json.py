import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from synonym_map import SYNONYM_MAP


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

    # ① 직접 매칭
    if text in SYNONYM_MAP:
        return SYNONYM_MAP[text]

    # ② 공백 제거 후 매칭
    text_nospace = text.replace(" ", "")
    if text_nospace in SYNONYM_MAP:
        return SYNONYM_MAP[text_nospace]

    # map에 없으면 그대로 반환
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


def steps_string_to_list(steps_str: str) -> list[str]:
    """
    "1. ...\n2. ...\n3. ..." 형태의 문자열을 리스트로 변환.
    번호 prefix(1. 2. 등)는 제거.
    """
    lines = steps_str.strip().split("\n")
    result = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # "1. " "2. " 같은 번호 prefix 제거
        line = re.sub(r"^\d+\.\s*", "", line)
        if line:
            result.append(line)
    return result


def convert_trending_recipes(input_path: str, output_path: str):
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    converted = []
    for recipe in data:
        converted.append({
            "id":                     recipe["id"],
            "title":                  recipe["title"],
            "ingredients":            recipe["ingredients"],
            "normalized_ingredients": build_normalized_ingredients(recipe["ingredients"]),
            "steps":                  steps_string_to_list(recipe["steps"]),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {output_path}")


if __name__ == "__main__":
    convert_trending_recipes(
        input_path=r"C:\VisionChef\recipes.json",
        output_path=r"C:\VisionChef\trending_recipes.json",
    )