from __future__ import annotations

import streamlit as st

try:
    from ..extraction import PDFDocument, select_pages
except ImportError:
    from extraction import PDFDocument, select_pages


def main() -> None:
    st.header("1. Extract images from a PDF")
    uploaded_pdf = st.file_uploader(
        "Select a PDF file",
        type=["pdf"],
        accept_multiple_files=False,
        max_upload_size=500,
    )

    if uploaded_pdf is None:
        return

    pdf_bytes = uploaded_pdf.getvalue()
    try:
        doc = PDFDocument(pdf_bytes, dpi=300, orientation_threshold=3.0)
        page_count = doc.page_count()
    except Exception as error:
        st.error(f"Could not read this PDF: {error}")
        return

    st.caption(f"{uploaded_pdf.name} - {page_count} page(s)")
    col_start, col_end, col_step = st.columns(3)
    with col_start:
        start = st.number_input(
            "Start page", min_value=1, max_value=page_count, value=1
        )
    with col_end:
        end = st.number_input(
            "End page (Included)", min_value=1, max_value=page_count, value=int(start)
        )
    with col_step:
        step = st.number_input("Step", min_value=1, max_value=page_count, value=1)

    if start > end:
        st.warning("Backward reading")

    pages = select_pages(int(start), int(end), int(step))

    st.write(f"Pages to extract: {', '.join(map(str, pages))}")

    if st.button("Confirm and extract images", type="primary"):
        with st.spinner("Rendering selected pages…"):
            try:
                images = doc.to_images(pages)
            except Exception as error:
                st.error(f"Could not extract the pages: {error}")
                return

        st.session_state.images = images
        st.session_state.pages = pages[: len(images)]
        st.session_state.ocr_images = None
        st.session_state.ocr_pages = None
        st.session_state.preprocess_steps = []
        st.session_state.selected_image = 0
        st.session_state.ocr_text = ""
        st.session_state.ocr_text_editor = ""
        st.session_state.annotated_image = None
        st.session_state.annotated_for = None
        st.session_state.annotated_config = None
        st.session_state.pdf_filename = uploaded_pdf.name

        # Reset all OCR page session state when a new document is extracted
        for key in list(st.session_state.keys()):
            if key.startswith("ocr_") or key in {
                "page_adjustment",
                "save_alert",
                "show_inline_overwrite_warning",
                "overwrite_approved",
            }:
                del st.session_state[key]

        st.success(f"Extracted {len(images)} image(s).")


if __name__ == "__main__":
    main()
