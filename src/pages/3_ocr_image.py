from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

try:
    from ..ocr_viet import (
        TesseractOCR,
    )
except ImportError:
    from ocr_viet import (
        TesseractOCR,
    )

OUTPUT_DIR = Path("Dong_hang")
LOW_CONFIDENCE_THRESHOLD = 85.0
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


def _image_to_pil(image: np.ndarray) -> Image.Image:
    """Convert an RGB or grayscale NumPy image to a PIL image."""
    if image.ndim == 2:
        return Image.fromarray(image)
    if image.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGBA2RGB))
    return Image.fromarray(image)


def _safe_filename(name: str) -> str:
    """Keep a user-provided text filename within the output directory."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    cleaned = Path(cleaned).name
    if not cleaned:
        cleaned = "ocr_result.txt"
    if not cleaned.lower().endswith(".txt"):
        cleaned += ".txt"
    return cleaned


def _bounding_box_filter_controls() -> tuple[int, int]:
    """Render the shared OCR bounding-box size controls in the sidebar."""
    st.subheader("Bounding-box filter")
    st.caption("Hide boxes when either dimension is below the selected size.")
    min_width = st.number_input(
        "Minimum box width (px)",
        min_value=0,
        value=10,
        step=1,
        key="bbox_min_width",
    )
    min_height = st.number_input(
        "Minimum box height (px)",
        min_value=0,
        value=10,
        step=1,
        key="bbox_min_height",
    )
    return int(min_width), int(min_height)


def _active_ocr_images() -> tuple[list[np.ndarray], list[int], bool]:
    """Return the current OCR inputs and whether they are preprocessed."""
    processed_images = st.session_state.get("ocr_images")
    if processed_images is not None:
        return (
            processed_images,
            st.session_state.get("ocr_pages") or st.session_state.get("pages", []),
            True,
        )
    return (
        st.session_state.get("images", []),
        st.session_state.get("pages", []),
        False,
    )


def main() -> None:
    st.header("3. OCR extracted images")
    images, pages, using_preprocessed = _active_ocr_images()
    with st.sidebar:
        st.divider()
        min_box_width, min_box_height = _bounding_box_filter_controls()

    if not images:
        st.info("Extract PDF pages on the first page before running OCR.")
        return

    if using_preprocessed:
        st.info("OCR is using the processed images from the preprocessing page.")
        if st.button("Restore original extracted images"):
            st.session_state.ocr_images = None
            st.session_state.ocr_pages = None
            st.session_state.annotated_image = None
            st.session_state.annotated_for = None
            st.rerun()

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
            or st.session_state.get("annotated_min_width") != min_box_width
            or st.session_state.get("annotated_min_height") != min_box_height
        ):
            display_image = _image_to_pil(current_image)

        st.image(
            display_image,
            width="stretch",
        )

        if st.button("Run OCR and draw bounding boxes", type="primary"):
            with st.spinner("Running OCR…"):
                try:
                    ocr = TesseractOCR(
                        lang="vie",
                        confidence_threshold=LOW_CONFIDENCE_THRESHOLD,
                    )
                    page = ocr.run(
                        current_image,
                        pdf_page_number=pages[image_index],
                        document_page_number=pages[image_index] + int(page_adjustment),
                        page_name=f"page_{pages[image_index]:04d}",
                        config=psm_config,
                        min_width=min_box_width,
                        min_height=min_box_height,
                    )
                    text = ocr.text
                    annotated = ocr.annotate_image()
                except Exception as error:
                    st.error(f"OCR failed: {error}")
                else:
                    st.session_state.ocr_text = text
                    st.session_state.ocr_text_editor = text
                    st.session_state.annotated_image = annotated
                    st.session_state.annotated_for = image_index
                    st.session_state.annotated_config = psm_config
                    st.session_state.annotated_min_width = min_box_width
                    st.session_state.annotated_min_height = min_box_height
                    st.session_state.ocr_page_obj = page
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
        col_save, col_dl_txt, col_dl_json = st.columns(3)
        with col_save:
            if st.button("Save text"):
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                destination = OUTPUT_DIR / safe_name
                destination.write_text(text, encoding="utf-8")
                st.success(f"Saved as {destination}")
        with col_dl_txt:
            st.download_button(
                "Download text",
                data=text,
                file_name=safe_name,
                mime="text/plain",
                disabled=not bool(text),
            )
        with col_dl_json:
            page_obj = st.session_state.get("ocr_page_obj")
            json_name = Path(safe_name).stem + ".json"
            st.download_button(
                "Download JSON",
                data=page_obj.model_dump_json(indent=2) if page_obj else "",
                file_name=json_name,
                mime="application/json",
                disabled=page_obj is None,
            )


if __name__ == "__main__":
    main()
