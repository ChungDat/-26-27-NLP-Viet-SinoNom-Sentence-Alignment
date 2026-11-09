from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytesseract
from PIL import Image, ImageDraw, ImageFont


OCRData = dict[str, list[Any]]


def get_ocr_data(
    image_path: str,
    config: str = "--psm 3",
    lang: str = "vie",
) -> OCRData:
    """Extract OCR text and bounding-box data from an image.

    Args:
        image_path: Path to the image to process.
        config: Tesseract page segmentation configuration.
        lang: Tesseract language code.

    Returns:
        OCR data containing recognized text, confidence scores, and bounding
        boxes for each detected item.
    """

    return pytesseract.image_to_data(
        image_path,
        output_type=pytesseract.Output.DICT,
        config=config,
        lang=lang,
    )


def get_ocr_text(
    image_path: str,
    config: str = "--psm 3",
    lang: str = "vie",
) -> str:
    """Extract text from an image using Tesseract.

    Args:
        image_path: Path to the image to process.
        config: Tesseract page segmentation configuration.
        lang: Tesseract language code.

    Returns:
        The raw text recognized in the image.
    """

    return pytesseract.image_to_string(
        image_path,
        config=config,
        lang=lang,
    )


def visualize(
    image_path: str,
    data: OCRData,
    destination_path: str,
    save_name: str,
) -> None:
    """Draw OCR bounding boxes and save the annotated image.

    Args:
        image_path: Path to the source image.
        data: OCR data returned by :func:`get_ocr_data`.
        destination_path: Path to the destination directory.
        save_name: Name of the annotated image file.
    """

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        25,
    )

    for index in range(len(data["text"])):
        recognized_text = str(data["text"][index]).strip()
        confidence_score = float(data["conf"][index])

        if not recognized_text or confidence_score < 0:
            continue

        left = int(data["left"][index])
        top = int(data["top"][index])
        width = int(data["width"][index])
        height = int(data["height"][index])

        draw.rectangle(
            (left, top, left + width, top + height),
            outline=(0, 255, 0),
            width=2,
        )

        label = (
            f"Block: {data['block_num'][index]}, Par: {data['par_num'][index]}, "
            f"Line: {data['line_num'][index]}, Word: {data['word_num'][index]}"
        )
        draw.text(
            (left, max(top - 30, 0)),
            label,
            font=font,
            fill=(255, 0, 0),
        )

    image.save(str(Path(destination_path) / save_name))


def normalize_text(text: str) -> str:
    """Put each sentence ending in a full stop on its own line.

    OCR may wrap one sentence over several lines, so single line breaks are
    first converted to spaces while paragraph breaks are preserved. A full
    stop followed by spaces or tabs then becomes the end of a line. A final
    full stop is also terminated with a newline.

    Args:
        text: OCR text to normalize.

    Returns:
        The normalized text with sentence-ending full stops on separate lines.
    """

    normalized_text = re.sub(
        r"\n+",
        lambda match: "\n" if len(match.group()) >= 2 else " ",
        text,
    )
    normalized_text = re.sub(r"\.[ \t]+", ".\n", normalized_text)
    return re.sub(r"\.(?=$)", ".\n", normalized_text)


def merge_box(data: OCRData) -> None:
    """Merge related OCR bounding boxes using heuristic rules.

    Args:
        data: OCR data containing the bounding boxes to merge.
    """

    pass


def save_text(text: str, destination_path: str, save_name: str) -> None:
    """Save OCR text to a file.

    Args:
        text: Text to save.
        destination_path: Path to the destination directory.
        save_name: Name of the output text file.
    """

    output_path = Path(destination_path) / save_name
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    """Run OCR on a sample image and save its outputs."""

    ocr_configs = ["--psm 3", "--psm 6"]
    source_directory = Path("Dong_hang/test")
    destination_directory = Path("Dong_hang/test")
    image_file_name = "0040.jpg"
    image_path = source_directory / "0" / image_file_name

    ocr_data = get_ocr_data(str(image_path), ocr_configs[0])
    visualize(str(image_path), ocr_data, str(destination_directory), "hypo.jpg")
    ocr_text = get_ocr_text(str(image_path), lang="vie")
    save_text(normalize_text(ocr_text), str(destination_directory), "hypo.txt")
    print(ocr_data)


if __name__ == "__main__":
    main()
