from __future__ import annotations

import streamlit as st

try:
    from .pages import (
        ocr_image,
        postprocess_image,
        preprocess_image,
        upload_image,
    )
except ImportError:
    from pages import (
        ocr_image,
        postprocess_image,
        preprocess_image,
        upload_image,
    )


def main() -> None:
    st.set_page_config(page_title="PDF OCR", layout="wide")

    if st.session_state.pop("open_ocr_after_apply", False):
        st.session_state.app_page = "Run OCR"

    st.title("PDF OCR workspace")

    page_map = {
        "Extract images": upload_image.main,
        "Preprocess": preprocess_image.main,
        "Run OCR": ocr_image.main,
        "Postprocess": postprocess_image.main,
    }

    page = st.sidebar.radio(
        "Page",
        list(page_map.keys()),
        key="app_page",
    )

    page_runner = page_map.get(page)
    if page_runner:
        page_runner()


if __name__ == "__main__":
    main()
