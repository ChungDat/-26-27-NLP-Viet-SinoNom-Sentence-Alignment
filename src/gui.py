from __future__ import annotations

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="PDF OCR", layout="wide")

    st.title("PDF OCR workspace")

    pages = [
        st.Page("pages/1_upload_document.py", title="Extract images"),
        st.Page("pages/2_preprocess_image.py", title="Preprocess"),
        st.Page("pages/3_ocr_viet.py", title="Run Viet OCR"),
        st.Page("pages/4_postprocess_image.py", title="Postprocess"),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
