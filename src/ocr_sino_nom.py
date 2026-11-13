import requests
import os
from dotenv import load_dotenv
from pathlib import Path

BASE_URL = "https://kimhannom.fit.hcmus.edu.vn"

load_dotenv()
token = os.getenv("TOKEN")

if not token:
    raise ValueError("TOKEN environment variable is missing")

header = {
    "Authorization": f"Bearer {token}",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0",
}


def upload_image(image_path: Path):
    endpoint = "/api/web/clc-sinonom/image-upload"

    with open(image_path, "rb") as image:
        response = requests.post(
            BASE_URL + endpoint,
            headers=header,
            files={
                "image_file": image
            }
        )

    return response


def classify_structure(image_name: str):
    endpoint = "/api/web/clc-sinonom/structure-classification"

    payload = {
        "file_name": image_name,
    }

    response = requests.post(
        BASE_URL + endpoint,
        headers=header,
        json=payload
    )

    return response


def ocr_image(image_name: str, ocr_id: int = 1, lang_type: int = 2, reading_direction: int = 1):
    endpoint = "/api/web/clc-sinonom/image-ocr"

    payload = {
        "ocr_id": ocr_id,
        "lang_type": lang_type,
        "reading_direction": reading_direction,
        "file_name": image_name
    }

    response = requests.post(
        BASE_URL + endpoint,
        headers=header,
        json=payload
    )

    return response


def rec_image(image_name: str, ocr_id: int = 1, lang_type: int = 2, reading_direction: int = 1):
    endpoint = "/api/web/clc-sinonom/image-rec"

    payload = {
        "ocr_id": ocr_id,
        "lang_type": lang_type,
        "reading_direction": reading_direction,
        "file_name": image_name
    }

    response = requests.post(
        BASE_URL + endpoint,
        headers=header,
        json=payload
    )

    return response


def check_response(response):
    response.raise_for_status()

    result = response.json()

    if not result.get("is_success"):
        raise RuntimeError(
            f"API failed:w {result}"
        )

    return result


def main():
    image_path = Path("Dong_hang/test/0/0459.jpg")

    upload_response = check_response(upload_image(image_path))
    temp_name = upload_response["data"]["file_name"]
    print(temp_name)
    structure_response = check_response(classify_structure(temp_name))
    ocr_response = check_response(ocr_image(temp_name))

    with open("api response.txt", "w") as file:
        file.write(repr(upload_response))
        file.write(repr(structure_response))
        file.write(repr(ocr_response))

if __name__ == "__main__":
    main()