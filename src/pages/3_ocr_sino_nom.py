from __future__ import annotations
import streamlit as st


def main() -> None:
    """Placeholder page for Sino-Nom OCR interface."""
    st.header("3. OCR Review — Sino-Nom")
    st.info("🚧 The Sino-Nom OCR interface is not yet implemented. Coming soon.")
    if st.button("← Back to Preprocess"):
        st.switch_page("pages/2_preprocess_image.py")


if __name__ == "__main__":
    main()
