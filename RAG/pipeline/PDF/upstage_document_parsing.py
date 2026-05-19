import json
from pathlib import Path
import requests


API_KEY = "up_T5brTUSN6x24WSVgtxv421FiPfxB7"   
PDF_PATH = r"C:\VisionChef\Baek_Book\4-4.pdf"
#r"C:\VisionChef\Baek_Book\***.pdf" 꼭 주의!!!!!!! 경로 설정 잘하자 숫자!
URL = "https://api.upstage.ai/v1/document-digitization"


def save_upstage_all_formats(
    api_key: str,
    pdf_path: str,
    model: str = "document-parse",
    ocr_force: bool = True,
) -> dict:
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

    # PDF 파일명 기준으로 출력 폴더 자동 생성
    output_dir = Path("upstage_outputs") / pdf_file.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    data = {
        "model": model,
    }

    if ocr_force:
        data["ocr"] = "force"

    with open(pdf_file, "rb") as f:
        files = {
            "document": (pdf_file.name, f, "application/pdf")
        }

        response = requests.post(
            URL,
            headers=headers,
            files=files,
            data=data,
            timeout=300,
        )

    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        print("요청 실패")
        print("status_code:", response.status_code)
        print("response_text:", response.text)
        raise e

    result = response.json()

    # 1) 응답 전체 JSON 저장
    raw_json_path = output_dir / "raw_response.json"
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    content = result.get("content", {})

    # 2) html 저장
    html_path = output_dir / "content.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(content.get("html", "") or "")

    # 3) markdown 저장
    md_path = output_dir / "content.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content.get("markdown", "") or "")

    # 4) text 저장
    txt_path = output_dir / "content.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(content.get("text", "") or "")

    # 5) elements만 따로 저장
    elements_path = output_dir / "elements.json"
    with open(elements_path, "w", encoding="utf-8") as f:
        json.dump(result.get("elements", []), f, ensure_ascii=False, indent=2)

    print("저장 완료:")
    print(f"- {raw_json_path}")
    print(f"- {html_path}")
    print(f"- {md_path}")
    print(f"- {txt_path}")
    print(f"- {elements_path}")

    return result


if __name__ == "__main__":
    save_upstage_all_formats(
        api_key=API_KEY,
        pdf_path=PDF_PATH,
        model="document-parse",
        ocr_force=True,
    )