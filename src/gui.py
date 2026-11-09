"""Streamlit interface for PDF page extraction and OCR."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from extraction import pdf_to_images, select_pages
from ocr_viet import get_ocr_data, get_ocr_text, normalize_text


OUTPUT_DIR = Path("outputs")
PSM_NOTES = {
    "--psm 3": "Default",
    "--psm 4": "Tabular-like data, each row is a sample (for table of content)",
    "--psm 6": "Uniform, single font face without any variation (for page without title)",
    "--psm 7": "Single text line",
    "--psm 8": "Single word",
    "--psm 10": "Single character",
    "--psm 11": "Sparse text, don't care about the order of sentences",
    "--psm 13": "Last resort",
}


def _pdf_page_count(pdf_bytes: bytes) -> int:
    """
    Return the number of pages in an uploaded PDF
    """

    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        return len(document)
    finally:
        document.close()


def _image_to_pil(image: np.ndarray) -> Image.Image:
    """
    Convert an OpenCV image to a PIL image for Streamlit display
    """
    
    if image.ndim == 2:
        return Image.fromarray(image)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _annotated_image(image: np.ndarray, data: dict) -> Image.Image:
    """
    Draw OCR word boxes over an image
    """

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)

    result = _image_to_pil(image).convert("RGB")
    draw = ImageDraw.Draw(result)

    for index, text in enumerate(data.get("text", [])):
        if not text.strip():
            continue
        try:
            confidence = float(data["conf"][index])
        except (KeyError, TypeError, ValueError):
            confidence = -1
        if confidence < 0:
            continue

        left = int(data["left"][index])
        top = int(data["top"][index])
        width = int(data["width"][index])
        height = int(data["height"][index])

        label = f"{text} ({confidence:.1f}%)"

        draw.rectangle(
            (left, top, left + width, top + height),
            outline=(0, 190, 0),
            width=2,
        )

        draw.text((left, max(top - 30, 0)), label, font=font, fill=(255, 0, 0))

    return result


def _ocr_image(image: np.ndarray, config: str) -> tuple[str, Image.Image]:
    """
    Run OCR on an in-memory image and return text plus an annotated image
    """

    with tempfile.NamedTemporaryFile(suffix=".jpg") as image_file:
        success, encoded = cv2.imencode(".jpg", image)
        if not success:
            raise ValueError("Could not encode the selected image")
        image_file.write(encoded.tobytes())
        image_file.flush()
        data = get_ocr_data(image_file.name, config=config)
        text = normalize_text(get_ocr_text(image_file.name, config=config))
    return text, _annotated_image(image, data)


def _safe_filename(name: str) -> str:
    """Keep a user-provided text filename within the output directory."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    cleaned = Path(cleaned).name
    if not cleaned:
        cleaned = "ocr_result.txt"
    if not cleaned.lower().endswith(".txt"):
        cleaned += ".txt"
    return cleaned


def _load_page() -> None:
    st.header("1. Extract images from a PDF")
    uploaded_pdf = st.file_uploader("Select a PDF file", type=["pdf"], accept_multiple_files=False, max_upload_size=500)

    if uploaded_pdf is None:
        return

    pdf_bytes = uploaded_pdf.getvalue()
    try:
        page_count = _pdf_page_count(pdf_bytes)
    except Exception as error:
        st.error(f"Could not read this PDF: {error}")
        return

    st.caption(f"{uploaded_pdf.name} - {page_count} page(s)")
    col_start, col_end, col_step = st.columns(3)
    with col_start:
        start = st.number_input("Start page", min_value=1, max_value=page_count, value=1)
    with col_end:
        end = st.number_input("End page (Included)", min_value=1, max_value=page_count, value=page_count)
    with col_step:
        step = st.number_input("Step", min_value=1, max_value=page_count, value=1)

    if start > end:
        st.warning("Backward reading")

    pages = select_pages(int(start), int(end), int(step))
    
    st.write(f"Pages to extract: {', '.join(map(str, pages))}")

    if st.button("Confirm and extract images", type="primary"):
        with st.spinner("Rendering selected pages…"):
            try:
                with tempfile.NamedTemporaryFile(suffix=".pdf") as pdf_file:
                    pdf_file.write(pdf_bytes)
                    pdf_file.flush()
                    images = pdf_to_images(pdf_file.name, pages, mode=0, dpi=300)
            except Exception as error:
                st.error(f"Could not extract the pages: {error}")
                return

        st.session_state.images = images
        st.session_state.pages = pages[: len(images)]
        st.session_state.selected_image = 0
        st.session_state.ocr_text = ""
        st.session_state.ocr_text_editor = ""
        st.session_state.annotated_image = None
        st.session_state.annotated_for = None
        st.session_state.annotated_config = None
        st.success(f"Extracted {len(images)} image(s). Open the OCR page to continue.")


def _ocr_page() -> None:
    st.header("2. OCR extracted images")
    images = st.session_state.get("images", [])
    pages = st.session_state.get("pages", [])
    if not images:
        st.info("Extract PDF pages on the first page before running OCR.")
        return

    if "ocr_text_editor" not in st.session_state:
        st.session_state.ocr_text_editor = st.session_state.get("ocr_text", "")

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        page_adjustment = st.number_input(
            "Adjust page",
            value=0,
            step=1,
            help=(
                "Offset between the PDF reader page number and the actual "
                "document page number. Actual page = PDF reader page + adjustment."
            ),
            key="page_adjustment",
        )
        image_index = st.selectbox(
            "Select an extracted image",
            range(len(images)),
            format_func=lambda index: (
                f"PDF reader page {pages[index]} "
                f"(actual page {pages[index] + int(page_adjustment)})"
            ),
            key="selected_image",
        )
        psm_config = st.selectbox(
            "Select page segmentation method",
            list(PSM_NOTES),
            format_func=lambda config: f"{config} - {PSM_NOTES[config]}",
            help="Choose the Tesseract page segmentation mode described in src/ocr_viet.py.",
            key="psm_config",
        )
        current_image = images[image_index]
        display_image = st.session_state.get("annotated_image")
        if (
            st.session_state.get("annotated_for") != image_index
            or st.session_state.get("annotated_config") != psm_config
        ):
            display_image = _image_to_pil(current_image)
            
        st.image(
            display_image,
            width="stretch",
        )

        if st.button("Run OCR and draw bounding boxes", type="primary"):
            with st.spinner("Running OCR…"):
                try:
                    text, annotated = _ocr_image(current_image, psm_config)
                except Exception as error:
                    st.error(f"OCR failed: {error}")
                else:
                    st.session_state.ocr_text = text
                    st.session_state.ocr_text_editor = text
                    st.session_state.annotated_image = annotated
                    st.session_state.annotated_for = image_index
                    st.session_state.annotated_config = psm_config
                    st.rerun()

    with right:
        text = st.text_area(
            "OCR extraction text",
            height=520,
            key="ocr_text_editor",
        )
        st.session_state.ocr_text = text
        filename = st.text_input("Text filename", value="ocr_result.txt")
        safe_name = _safe_filename(filename)
        if st.button("Save text"):
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            destination = OUTPUT_DIR / safe_name
            destination.write_text(text, encoding="utf-8")
            st.success(f"Saved as {destination}")
        st.download_button(
            "Download text",
            data=text,
            file_name=safe_name,
            mime="text/plain",
            disabled=not bool(text),
        )


def main() -> None:
    st.set_page_config(page_title="PDF OCR", layout="wide")
    st.title("PDF OCR workspace")
    page = st.sidebar.radio("Page", ["Extract images", "Run OCR"])
    if page == "Extract images":
        _load_page()
    else:
        _ocr_page()


if __name__ == "__main__":
    main()
