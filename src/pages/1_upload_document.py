from __future__ import annotations

from pathlib import Path

import cv2
import streamlit as st

try:
    from ..extraction import PDFDocument, select_pages
except ImportError:
    from extraction import PDFDocument, select_pages

MAX_UPLOAD_SIZE = 500  # MB
ORIENTATION_THRESHOLD = 3.0


def select_type(type):
    if type == "vi":
        return "Quốc ngữ (vi)"
    elif type == "ch":
        return "Hán Nôm (ch)"
    else:
        return "Khác"


def main() -> None:
    st.header("1. Extract images from a PDF")
    uploaded_pdf = st.file_uploader(
        "Select a PDF file",
        type=["pdf"],
        accept_multiple_files=False,
        max_upload_size=MAX_UPLOAD_SIZE,
    )

    if uploaded_pdf is None:
        return

    pdf_bytes = uploaded_pdf.getvalue()
    try:
        doc = PDFDocument(pdf_bytes, orientation_threshold=ORIENTATION_THRESHOLD)
        page_count = doc.page_count()
    except (ValueError, FileNotFoundError) as error:
        st.error(f"Could not read this PDF: {error}")
        return

    st.caption(f"{uploaded_pdf.name} - {page_count} page(s)")
    col_start, col_end, col_step, col_lang = st.columns(4)
    with col_start:
        start = st.number_input(
            "Start page", min_value=1, max_value=page_count, value=1
        )
    with col_end:
        end = st.number_input(
            "End page (Inclusive)", min_value=1, max_value=page_count, value=int(start)
        )
    with col_step:
        step = st.number_input("Step", min_value=1, max_value=page_count, value=1)
    with col_lang:
        document_type = st.radio(  # only for routing destination path, does not related to 2_preprocess_image and 3_ocr_viet language
            "Document type",
            options=["vi", "ch", "extra"],
            index=0,
            format_func=select_type,
            horizontal=True,
        )

    if start > end:
        st.warning("Backward reading")

    try:
        pages = select_pages(int(start), int(end), int(step))
    except ValueError as error:
        st.error(f"Invalid page selection: {error}")
        return

    st.write(f"Pages to extract: {', '.join(map(str, pages))}")

    if st.button("Confirm and extract images", type="primary"):
        document_name = Path(uploaded_pdf.name).stem
        destination_dir = Path(document_name) / document_type

        with st.spinner("Extracting embedded images from selected pages…"):
            try:
                doc.save_images(
                    pdf_pages=pages,
                    document_name=document_name,
                    document_type=document_type,
                )
            except (ValueError, FileNotFoundError, RuntimeError) as error:
                st.error(f"Could not extract images: {error}")
                return

        # Load the extracted images from disk into session state
        loaded_pages = []
        for page_num in pages:
            pattern = f"{document_name}.{page_num:03d}*.jpg"
            matched_files = sorted(destination_dir.glob(pattern))
            for file_path in matched_files:
                bgr_img = cv2.imread(str(file_path))
                if bgr_img is not None:
                    loaded_pages.append(page_num)

        st.session_state.pages = loaded_pages
        st.session_state.ocr_images = None
        st.session_state.ocr_pages = None
        st.session_state.preprocess_steps = []
        st.session_state.selected_image = 0
        st.session_state.ocr_text = ""
        st.session_state.ocr_text_editor = ""
        st.session_state.annotated_image = None
        st.session_state.annotated_for = None
        st.session_state.annotated_config = None
        st.session_state.document_name = document_name

        # Reset all OCR page session state when a new document is extracted
        for key in list(st.session_state.keys()):
            if key.startswith("ocr_") or key in {
                "page_adjustment",
                "save_alert",
                "show_inline_overwrite_warning",
                "overwrite_approved",
            }:
                del st.session_state[key]

        st.success(
            f"Extracted and saved {len(loaded_pages)}/{len(pages)} image(s) to '{destination_dir}'."
        )


if __name__ == "__main__":
    main()
