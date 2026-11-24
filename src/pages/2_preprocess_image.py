from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from ..ocr_viet import VLMOCR, TesseractOCR
    from ..preprocess import PreprocessPipeline
except ImportError:
    from ocr_viet import VLMOCR, TesseractOCR  # type: ignore[no-redef]
    from preprocess import PreprocessPipeline  # type: ignore[no-redef]

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


def _text_stats(text: str) -> dict[str, int]:
    """Calculate character, word, and paragraph count for a given text."""
    stripped = text.strip()
    if not stripped:
        return {"chars": 0, "words": 0, "paragraphs": 0}
    return {
        "chars": len(stripped),
        "words": len(stripped.split()),
        "paragraphs": len([ln for ln in stripped.splitlines() if ln.strip()]),
    }


def _set_comparison_text(version: int, text: str) -> None:
    """Keep comparison results and their text-area widget values in sync."""
    st.session_state[f"cmp_v{version}_text"] = text
    st.session_state[f"cmp_v{version}_area"] = text
    st.session_state[f"v{version}_col_area"] = text


def _clear_comparison_state() -> None:
    """Clear all extraction results and metadata across comparison versions."""
    for version in range(1, 5):
        _set_comparison_text(version, "")
    st.session_state.cmp_v1_annotated = None
    st.session_state.cmp_v3_annotated = None
    st.session_state.cmp_v2_response = None
    st.session_state.cmp_v4_response = None
    st.session_state.cmp_v1_status = "Not run"
    st.session_state.cmp_v2_status = "Not run"
    st.session_state.cmp_v3_status = "Not run"
    st.session_state.cmp_v4_status = "Not run"


def _uploaded_image(uploaded_file: Any) -> np.ndarray | None:
    if uploaded_file is None:
        return None
    try:
        return np.asarray(Image.open(uploaded_file).convert("RGB"))
    except Exception as error:
        st.error(f"Could not read the uploaded image: {error}")
        return None


def _render_step_controls(pipeline: PreprocessPipeline, index: int) -> bool:
    """Dynamically render controls for one pipeline step based on its parameter specs."""
    step = pipeline.steps[index]
    step_id = getattr(step, "id", f"step-{index}")
    step_label = getattr(step, "label", getattr(step, "name", f"Step {index + 1}"))
    params = getattr(step, "params", {})
    changed = False

    with st.expander(f"{index + 1}. {step_label}", expanded=True):
        description = getattr(step, "description", None)
        if description:
            st.caption(description)

        for spec in getattr(step, "param_specs", ()):
            # Check conditional visibility
            if spec.visible_when:
                if any(params.get(k) != v for k, v in spec.visible_when.items()):
                    continue

            current_val = params.get(spec.name, spec.default)
            widget_key = f"{step_id}-{spec.name}"

            if spec.type == "select":
                options = spec.options or []
                fmt = (
                    (
                        lambda value, labels=spec.option_labels: labels.get(
                            value, str(value)
                        )
                    )
                    if spec.option_labels
                    else str
                )
                opt_idx = options.index(current_val) if current_val in options else 0
                new_val = st.selectbox(
                    spec.label,
                    options,
                    index=opt_idx,
                    format_func=fmt,
                    help=spec.help,
                    key=widget_key,
                )
            elif spec.type == "slider":
                new_val = st.slider(
                    spec.label,
                    min_value=spec.min_value,
                    max_value=spec.max_value,
                    value=current_val,
                    step=spec.step or 1,
                    help=spec.help,
                    key=widget_key,
                )
            elif spec.type == "number":
                new_val = st.number_input(
                    spec.label,
                    min_value=spec.min_value,
                    max_value=spec.max_value,
                    value=current_val,
                    step=spec.step or 1,
                    help=spec.help,
                    key=widget_key,
                )
            else:
                continue

            if params.get(spec.name) != new_val:
                params[spec.name] = new_val
                changed = True

        move_left, move_right, remove = st.columns(3)
        with move_left:
            if st.button("Move up", key=f"{step_id}-up", disabled=index == 0):
                pipeline.move_step(index, index - 1)
                return True
        with move_right:
            if st.button(
                "Move down",
                key=f"{step_id}-down",
                disabled=index == len(pipeline.steps) - 1,
            ):
                pipeline.move_step(index, index + 1)
                return True
        with remove:
            if st.button("Remove", key=f"{step_id}-remove"):
                pipeline.remove_step(index)
                return True

    return changed


def _run_tesseract_extraction(
    image: np.ndarray,
    config: str,
    min_width: int,
    min_height: int,
) -> tuple[str, Any, str]:
    """Execute Tesseract OCR and return (text, annotated_image, status)."""
    try:
        tess = TesseractOCR(lang="vie", confidence_threshold=LOW_CONFIDENCE_THRESHOLD)
        tess.run(image, config=config, min_width=min_width, min_height=min_height)
        return tess.text, tess.annotate_image(), "Success"
    except Exception as error:
        return f"Error: {error}", None, f"Failed ({error})"


def _run_vlm_extraction(
    image: np.ndarray,
    model: str,
    temperature: float,
) -> tuple[str, Any, str]:
    """Execute VLM OCR and return (text, raw_response, status)."""
    try:
        vlm = VLMOCR(model=model, options={"temperature": temperature})
        vlm.run(image)
        return vlm.text, vlm.raw_response, "Success"
    except Exception as error:
        return f"Error: {error}", None, f"Failed ({error})"


def main() -> None:
    """Main entry point for the Preprocess Images page (page 2)."""
    st.header("2. Preprocess images")

    # Language selector to route to different ocr pipeline
    col_language, col_source = st.columns(2)
    with col_language:
        language = st.radio(
            "Language",
            options=["vi", "ch"],
            index=0,
            format_func=lambda x: "Quốc ngữ (vi)" if x == "vi" else "Hán Nôm (ch)",
            horizontal=True,
            key="document_language",
        )

    with col_source:
        document_name = st.text_area(
            "Source (PDF filename)", key="document_name"
        ).strip()

    if not document_name:
        st.warning(
            "⚠️ No document specified. Extract a document on Page 1 "
            "(1_upload_document) or enter the document name above."
        )
        st.stop()

    doc_dir = Path(document_name) / language
    if not doc_dir.is_dir():
        st.warning(
            f"⚠️ Extracted images directory `{doc_dir}` not found. "
            f"Extract pages for '{document_name}' on Page 1 (1_upload_document) "
            f"or check that the language ('{language}') and document name match your extracted folder."
        )
        st.stop()

    extracted_pages = []
    for entry in sorted(doc_dir.glob("*.jpg")):
        file_name = entry.stem
        parts = file_name.split(".")
        if len(parts) >= 2:
            extracted_pages.append(parts[-1])

    if not extracted_pages:
        st.warning(
            f"⚠️ No image files found in `{doc_dir}`. "
            "Extract images on Page 1 (1_upload_document) first."
        )
        st.stop()

    if "preprocess_steps" not in st.session_state:
        st.session_state.preprocess_steps = []
    pipeline = PreprocessPipeline(st.session_state.preprocess_steps)

    # Track whether modifications have been applied
    st.session_state["preprocess_modified"] = bool(pipeline.steps)

    with st.sidebar:
        source_options = ["Upload an image"] + [
            f"Extracted page {page_num}" for page_num in extracted_pages
        ]
        source_index = st.selectbox(
            "Source image",
            range(len(source_options)),
            format_func=source_options.__getitem__,
            key="preprocess_source_index",
        )

        if source_index == 0:
            uploaded_image = st.file_uploader(
                "Upload an image",
                type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
                key="preprocess_uploaded_image",
            )
            source_image = _uploaded_image(uploaded_image)
        else:
            page_num = extracted_pages[source_index - 1]
            file_path = (
                Path(document_name) / language / f"{document_name}.{page_num}.jpg"
            )
            if not file_path.exists():
                matched = sorted(
                    (Path(document_name) / language).glob(
                        f"{document_name}.{page_num}*.jpg"
                    )
                )
                if matched:
                    file_path = matched[0]
            bgr_img = cv2.imread(str(file_path))
            if bgr_img is not None:
                source_image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        st.subheader("Transformation pipeline")

        # Dynamically discover available transformations from PreprocessPipeline
        available_transforms = PreprocessPipeline.get_available_transformations()
        add_type = st.selectbox(
            "Transformation to add",
            options=list(available_transforms.keys()),
            format_func=lambda key: available_transforms[key].label,
            key="preprocess_add_type",
        )

        add_col, reset_col = st.columns(2)
        with add_col:
            if st.button("Add transformation"):
                pipeline.add_step(add_type)
                st.session_state.preprocess_steps = pipeline.to_dict()
                st.session_state["preprocess_modified"] = bool(pipeline.steps)
                st.rerun()
        with reset_col:
            if st.button("Reset pipeline"):
                pipeline = PreprocessPipeline()
                st.session_state.preprocess_steps = []
                st.session_state["preprocess_modified"] = False
                _clear_comparison_state()
                st.rerun()

        for index in range(len(pipeline.steps)):
            if _render_step_controls(pipeline, index):
                st.session_state.preprocess_steps = pipeline.to_dict()
                st.session_state["preprocess_modified"] = bool(pipeline.steps)
                st.rerun()

        st.session_state.preprocess_steps = pipeline.to_dict()
        st.session_state["preprocess_modified"] = bool(pipeline.steps)

        settings = {"version": 1, "steps": pipeline.to_dict()}
        st.download_button(
            "Save preprocess settings",
            data=json.dumps(settings, indent=2),
            file_name="preprocess_settings.json",
            mime="application/json",
            disabled=not pipeline.steps,
            help="Download the current transformation pipeline as JSON.",
        )

        imported_settings = st.file_uploader(
            "Saved preprocess settings",
            type=["json"],
            key="preprocess_settings_upload",
            accept_multiple_files=False,
        )
        if st.button(
            "Import preprocess settings",
            disabled=imported_settings is None,
        ):
            try:
                raw_data = json.loads(imported_settings.getvalue().decode("utf-8"))
                imported_pipeline = PreprocessPipeline.from_dict(raw_data)
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                ValueError,
                TypeError,
            ) as error:
                st.error(f"Could not import settings: {error}")
            else:
                st.session_state.preprocess_steps = imported_pipeline.to_dict()
                st.session_state["preprocess_modified"] = bool(imported_pipeline.steps)
                _clear_comparison_state()
                st.rerun()

    if source_image is None:
        st.info("Choose or upload an image to preview the pipeline.")
        return

    try:
        processed_image = pipeline.apply(source_image)
    except (TypeError, ValueError, cv2.error) as error:
        st.error(f"Could not apply the pipeline: {error}")
        return

    original_col, processed_col = st.columns(2)
    with original_col:
        st.image(_image_to_pil(source_image), caption="Original", width="stretch")
    with processed_col:
        st.image(_image_to_pil(processed_image), caption="Processed", width="stretch")

    st.divider()
    st.subheader("Extraction Comparison: Tesseract vs VLM (4 versions)")
    st.caption(
        "Compare OCR text extractions between original and preprocessed images "
        "using both Tesseract and Vision-Language Model (VLM)."
    )

    with st.expander("Comparison engine settings", expanded=False):
        cfg_col1, cfg_col2 = st.columns(2)
        with cfg_col1:
            st.markdown("**Tesseract settings**")
            cmp_psm = st.selectbox(
                "Page segmentation method (PSM)",
                list(PSM_NOTES),
                format_func=lambda config: f"{config} - {PSM_NOTES[config]}",
                key="cmp_psm_config",
            )

            min_box_width = st.number_input(
                "Minimum box width (px)",
                min_value=0,
                value=5,
                step=1,
                key="bbox_min_width",
            )
            min_box_height = st.number_input(
                "Minimum box height (px)",
                min_value=0,
                value=5,
                step=1,
                key="bbox_min_height",
            )

        with cfg_col2:
            st.markdown("**VLM (Ollama) settings**")
            vlm_model = st.text_input(
                "Ollama VLM model",
                value="qwen3-vl:2b-instruct",
                help="Multimodal vision model name in Ollama.",
                key="cmp_vlm_model",
            )
            vlm_temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                key="cmp_vlm_temp",
            )

    btn_all, btn_tess, btn_vlm = st.columns([1.5, 1, 1])
    run_all = btn_all.button("⚡ Run 4-Way Comparison")
    run_tess = btn_tess.button("Run Tesseract extractions (V1 & V3)")
    run_vlm = btn_vlm.button("Run VLM extractions (V2 & V4)")

    if run_all or run_tess:
        with st.spinner("Running Tesseract OCR on original & preprocessed images…"):
            v1_text, v1_ann, v1_st = _run_tesseract_extraction(
                source_image, cmp_psm, min_box_width, min_box_height
            )
            _set_comparison_text(1, v1_text)
            st.session_state.cmp_v1_annotated = v1_ann
            st.session_state.cmp_v1_status = v1_st
            if v1_st != "Success":
                st.error(f"Tesseract (Original): {v1_text}")

            v3_text, v3_ann, v3_st = _run_tesseract_extraction(
                processed_image, cmp_psm, min_box_width, min_box_height
            )
            _set_comparison_text(3, v3_text)
            st.session_state.cmp_v3_annotated = v3_ann
            st.session_state.cmp_v3_status = v3_st
            if v3_st != "Success":
                st.error(f"Tesseract (Preprocessed): {v3_text}")

    if run_all or run_vlm:
        with st.spinner("Running VLM OCR on original & preprocessed images…"):
            v2_text, v2_resp, v2_st = _run_vlm_extraction(
                source_image, vlm_model, vlm_temperature
            )
            _set_comparison_text(2, v2_text)
            st.session_state.cmp_v2_response = v2_resp
            st.session_state.cmp_v2_status = v2_st
            if v2_st != "Success":
                st.warning(f"VLM (Original): {v2_text}")

            v4_text, v4_resp, v4_st = _run_vlm_extraction(
                processed_image, vlm_model, vlm_temperature
            )
            _set_comparison_text(4, v4_text)
            st.session_state.cmp_v4_response = v4_resp
            st.session_state.cmp_v4_status = v4_st
            if v4_st != "Success":
                st.warning(f"VLM (Preprocessed): {v4_text}")

    v1_text = st.session_state.get("cmp_v1_text", "")
    v2_text = st.session_state.get("cmp_v2_text", "")
    v3_text = st.session_state.get("cmp_v3_text", "")
    v4_text = st.session_state.get("cmp_v4_text", "")
    v1_annotated = st.session_state.get("cmp_v1_annotated")
    v3_annotated = st.session_state.get("cmp_v3_annotated")

    has_extractions = any([v1_text, v2_text, v3_text, v4_text])
    if has_extractions:
        tab_side, tab_summary = st.tabs(["Side-by-Side View", "Metrics Summary"])

        with tab_side:
            col_orig, col_proc = st.columns(2, gap="medium")
            with col_orig:
                st.markdown("#### 📄 Original Image Extractions")
                st.markdown("**Version 1: Tesseract OCR**")
                if v1_annotated is not None:
                    st.image(
                        v1_annotated,
                        caption="Original - Tesseract word boxes",
                        width="stretch",
                    )
                s1 = _text_stats(v1_text)
                st.caption(
                    f"{s1['chars']} chars | {s1['words']} words | {s1['paragraphs']} paragraphs"
                )
                st.text_area(
                    "V1 Text (Original + Tesseract)",
                    value=v1_text,
                    height=220,
                    key="cmp_v1_area",
                )

                st.markdown("**Version 2: VLM OCR**")
                s2 = _text_stats(v2_text)
                v2_meta = ""
                resp2 = st.session_state.get("cmp_v2_response")
                if resp2 and hasattr(resp2, "eval_count") and resp2.eval_count:
                    v2_meta = f" | {resp2.eval_count} tokens"
                st.caption(
                    f"{s2['chars']} chars | {s2['words']} words | {s2['paragraphs']} paragraphs{v2_meta}"
                )
                st.text_area(
                    "V2 Text (Original + VLM)",
                    value=v2_text,
                    height=220,
                    key="cmp_v2_area",
                )

            with col_proc:
                st.markdown("#### ⚙️ Preprocessed Image Extractions")
                st.markdown("**Version 3: Tesseract OCR**")
                if v3_annotated is not None:
                    st.image(
                        v3_annotated,
                        caption="Preprocessed - Tesseract word boxes",
                        width="stretch",
                    )
                s3 = _text_stats(v3_text)
                st.caption(
                    f"{s3['chars']} chars | {s3['words']} words | {s3['paragraphs']} paragraphs"
                )
                st.text_area(
                    "V3 Text (Preprocessed + Tesseract)",
                    value=v3_text,
                    height=220,
                    key="cmp_v3_area",
                )

                st.markdown("**Version 4: VLM OCR**")
                s4 = _text_stats(v4_text)
                v4_meta = ""
                resp4 = st.session_state.get("cmp_v4_response")
                if resp4 and hasattr(resp4, "eval_count") and resp4.eval_count:
                    v4_meta = f" | {resp4.eval_count} tokens"
                st.caption(
                    f"{s4['chars']} chars | {s4['words']} words | {s4['paragraphs']} paragraphs{v4_meta}"
                )
                st.text_area(
                    "V4 Text (Preprocessed + VLM)",
                    value=v4_text,
                    height=220,
                    key="cmp_v4_area",
                )

        with tab_summary:
            summary_data = [
                {
                    "Version": f"V{v}",
                    "Source Image": src,
                    "OCR Engine": eng,
                    "Characters": _text_stats(txt)["chars"],
                    "Words": _text_stats(txt)["words"],
                    "Paragraphs": _text_stats(txt)["paragraphs"],
                    "Status": st.session_state.get(f"cmp_v{v}_status", "Not run"),
                }
                for v, src, eng, txt in [
                    (1, "Original", "Tesseract", v1_text),
                    (2, "Original", "VLM", v2_text),
                    (3, "Preprocessed", "Tesseract", v3_text),
                    (4, "Preprocessed", "VLM", v4_text),
                ]
            ]
            st.dataframe(pd.DataFrame(summary_data), width="stretch")
        st.info(
            "Click '⚡ Run 4-Way Comparison' to run extractions on both images with both engines."
        )

    if st.button("Apply pipeline to all extracted images and open OCR", type="primary"):
        st.session_state.preprocess_steps = pipeline.to_dict()
        st.session_state["preprocess_modified"] = bool(pipeline.steps)

        # Clear OCR session state so page 3 starts fresh with new pipeline
        for key in [
            "ocr_current_index",
            "ocr_page_cache",
            "ocr_images_signature",
            "ocr_confirmed_flags",
            "ocr_confirmed_pages",
            "page_adjustment",
            "save_alert",
            "show_inline_overwrite_warning",
            "overwrite_approved",
            "ocr_preprocess_signature",
            "ocr_images",
            "ocr_pages",
        ]:
            st.session_state.pop(key, None)
        for key in list(st.session_state.keys()):
            if key.startswith(
                ("ocr_editor_widget_", "ocr_vlm_widget_", "image_view_mode_")
            ):
                del st.session_state[key]

        target_page = (
            "pages/3_ocr_viet.py"
            if (
                (Path(__file__).parent / "3_ocr_viet.py").exists()
                or Path("src/pages/3_ocr_viet.py").exists()
                or Path("pages/3_ocr_viet.py").exists()
            )
            else "pages/3_ocr_image.py"
        )
        if language == "vi":
            st.switch_page(target_page)
        else:
            st.switch_page("pages/3_ocr_sino_nom.py")


if __name__ == "__main__":
    main()
