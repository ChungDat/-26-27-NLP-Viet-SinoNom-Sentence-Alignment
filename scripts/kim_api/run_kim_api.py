import getpass
import json
import time
from pathlib import Path
from typing import Any

import requests


BASE = "https://kimhannom.fit.hcmus.edu.vn"

# Script location: <repo>/scripts/kim_api/run_kim_api_dual.py
ROOT = Path(__file__).resolve().parents[2]

# Input is the Sino-Nom page; its paired Quốc ngữ page is not sent to the API.
IMAGE = ROOT / "pilot" / "images" / "han_nom" / "hn_001.png"

OUT = ROOT / "pilot" / "api_output"
OUT.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0"
    ),
    "Authorization": "Bearer " + getpass.getpass("Nhap API token: "),
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def check_response(response: requests.Response) -> dict[str, Any]:
    response.raise_for_status()
    envelope = response.json()

    if envelope.get("is_success") is not True or envelope.get("code") != "000000":
        raise RuntimeError(json.dumps(envelope, ensure_ascii=False, indent=2))

    data = envelope.get("data")
    if not isinstance(data, dict):
        raise RuntimeError(
            "API báo thành công nhưng data không phải object:\n"
            + json.dumps(envelope, ensure_ascii=False, indent=2)
        )
    return data


def find_text_fields(value: Any, prefix: str = "data") -> list[tuple[str, Any]]:
    """Collect text-like fields recursively without assuming undocumented names."""
    found: list[tuple[str, Any]] = []

    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}"
            key_lower = str(key).lower()
            if isinstance(child, str) and child.strip():
                found.append((path, child))
            elif (
                isinstance(child, list)
                and child
                and all(isinstance(item, str) for item in child)
            ):
                found.append((path, child))
            else:
                found.extend(find_text_fields(child, path))

    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(find_text_fields(child, f"{prefix}[{index}]"))

    return found


def as_text(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value)


def main() -> None:
    if not IMAGE.is_file():
        raise FileNotFoundError(
            f"Khong tim thay anh dau vao: {IMAGE}\n"
            "Kiem tra lai IMAGE trong run_kim_api_dual.py."
        )

    stem = IMAGE.stem
    print(f"Anh dau vao: {IMAGE}")
    print("Loai anh: Hán Nôm, văn bản dọc, đọc từ phải sang trái.")
    print("Script sẽ gọi upload và image-ocr đúng một lần mỗi endpoint.")

    # 1. Upload image
    print("\n[1/2] UPLOAD IMAGE")
    with IMAGE.open("rb") as image_file:
        uploaded = check_response(
            SESSION.post(
                BASE + "/api/web/clc-sinonom/image-upload",
                files={"image_file": (IMAGE.name, image_file, "image/png")},
                timeout=120,
            )
        )

    file_name = uploaded["file_name"]
    print("Upload thành công:", file_name)

    # 2. OCR Sino-Nom. Keep the known working parameters from the previous test.
    time.sleep(2)
    print("\n[2/2] IMAGE OCR")
    payload = {
        "ocr_id": 1,
        "lang_type": 2,
        "reading_direction": 1,
        "file_name": file_name,
    }
    print("OCR payload:", json.dumps(payload, ensure_ascii=False))

    ocr_data = check_response(
        SESSION.post(
            BASE + "/api/web/clc-sinonom/image-ocr",
            json=payload,
            timeout=180,
        )
    )

    # Always preserve the complete response; this is the source of truth
    # for discovering any additional Quốc ngữ field returned by the server.
    raw_path = OUT / f"{stem}_response.json"
    raw_path.write_text(
        json.dumps(ocr_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # The documented and previously working Hán-Nôm field.
    hn_value = ocr_data.get("result_ocr_text")
    if hn_value is not None:
        (OUT / f"{stem}_han_nom.txt").write_text(as_text(hn_value), encoding="utf-8")

    # The API guide documents result_ocr_text for the selected OCR language,
    # but does not specify a Quốc ngữ field. Do not guess a field name:
    # save every other text-like field separately for inspection.
    text_fields = find_text_fields(ocr_data)
    field_report_lines = []
    qn_candidates: list[tuple[str, Any]] = []

    qn_markers = (
        "quoc_ngu",
        "quocngu",
        "vietnamese",
        "tieng_viet",
        "text_vi",
        "vi_text",
        "ocr_vi",
        "result_vi",
        "_vi",
        "quocngu_text",
        "qn_text",
    )

    for path, value in text_fields:
        lower_path = path.lower()
        field_report_lines.append(f"===== {path} =====\n{as_text(value)}\n")
        if path != "data.result_ocr_text" and any(
            marker in lower_path for marker in qn_markers
        ):
            qn_candidates.append((path, value))

    (OUT / f"{stem}_text_fields.txt").write_text(
        "\n".join(field_report_lines)
        if field_report_lines
        else "Không tìm thấy trường văn bản dạng string/list[string].\n",
        encoding="utf-8",
    )

    if len(qn_candidates) == 1:
        qn_path, qn_value = qn_candidates[0]
        (OUT / f"{stem}_quoc_ngu.txt").write_text(as_text(qn_value), encoding="utf-8")
        print(f"\nĐã nhận diện trường Quốc ngữ theo tên trường: {qn_path}")
    elif len(qn_candidates) > 1:
        print(
            "\nCó nhiều trường có tên gợi ý Quốc ngữ; chưa tự chọn để tránh gán nhầm:"
        )
        for path, _ in qn_candidates:
            print(" -", path)
    else:
        print(
            "\nChưa thấy trường có tên thể hiện rõ Quốc ngữ. "
            "Không kết luận API không trả Quốc ngữ; hãy kiểm tra "
            f"{stem}_response.json và {stem}_text_fields.txt."
        )

    print("\nOCR SUCCESS")
    print("Các trường data:", ", ".join(ocr_data.keys()))
    if isinstance(hn_value, list):
        print("Số dòng Hán Nôm:", len(hn_value))
    print("Đã lưu JSON đầy đủ:", raw_path)
    print("Thư mục kết quả:", OUT.resolve())


if __name__ == "__main__":
    main()
