from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from ..ocr_viet import (
        VLMOCR,
        TesseractOCR,
    )
    from ..preprocess import PreprocessPipeline
except ImportError:
    from ocr_viet import (
        VLMOCR,
        TesseractOCR,
    )
    from preprocess import PreprocessPipeline

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


def _bounding_box_filter_controls() -> tuple[int, int]:
    """Render the shared OCR bounding-box size controls in the sidebar."""
    st.subheader("Bounding-box filter")
    st.caption("Hide boxes when either dimension is below the selected size.")
    min_width = st.number_input(
        "Minimum box width (px)",
        min_value=0,
        value=50,
        step=1,
        key="bbox_min_width",
    )
    min_height = st.number_input(
        "Minimum box height (px)",
        min_value=0,
        value=50,
        step=1,
        key="bbox_min_height",
    )
    return int(min_width), int(min_height)


def _uploaded_image(uploaded_file) -> np.ndarray | None:
    if uploaded_file is None:
        return None
    try:
        return np.asarray(Image.open(uploaded_file).convert("RGB"))
    except Exception as error:
        st.error(f"Could not read the uploaded image: {error}")
        return None


def _step_label(step_type: str) -> str:
    return {
        "convert_color": "Convert color",
        "reverse_color": "Reverse color",
        "filter": "Filter (median)",
        "binary_morphology": "Binary morphology",
        "grayscale_morphology": "Grayscale morphology",
    }.get(step_type, step_type.replace("_", " ").title())


def _pipeline_from_settings(settings: object) -> PreprocessPipeline:
    """Validate a saved JSON settings object and build its pipeline."""
    if not isinstance(settings, dict) or settings.get("version") != 1:
        raise ValueError("settings must be a version 1 JSON object")
    steps = settings.get("steps")
    if not isinstance(steps, list):
        raise ValueError("settings must contain a list of pipeline steps")

    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict) or not isinstance(step.get("type"), str):
            raise ValueError(f"step {index} is missing its transformation type")
        if "params" in step and not isinstance(step["params"], dict):
            raise ValueError(f"step {index} has invalid transformation parameters")
        PreprocessPipeline.default_step(step["type"])
    return PreprocessPipeline(steps)


def _render_step_controls(pipeline: PreprocessPipeline, index: int) -> bool:
    """Render one editable pipeline step and return whether it was changed."""
    step = pipeline.steps[index]
    step_id = step.setdefault("id", f"step-{index}")
    step_type = step.get("type", "")
    params = step.setdefault("params", {})
    changed = False

    with st.expander(f"{index + 1}. {_step_label(step_type)}", expanded=True):
        if step_type == "convert_color":
            mode_labels = {0: "RGB", 1: "Grayscale", 2: "Binary (Otsu)"}
            current_mode = int(params.get("mode", 1))
            mode = st.selectbox(
                "Output mode",
                list(mode_labels),
                format_func=mode_labels.get,
                index=list(mode_labels).index(current_mode)
                if current_mode in mode_labels
                else 1,
                key=f"{step_id}-mode",
            )
            if params.get("mode") != mode:
                params["mode"] = mode
                changed = True
        elif step_type == "reverse_color":
            st.caption("Invert all channel values, exchanging black and white.")
        elif step_type == "filter":
            kernel_size = st.slider(
                "Median kernel size",
                min_value=3,
                max_value=31,
                value=int(params.get("kernel_size", 3)),
                step=2,
                help="Larger values remove more salt-and-pepper artifacts.",
                key=f"{step_id}-filter-kernel-size",
            )
            if params.get("kernel_size") != kernel_size:
                params["kernel_size"] = kernel_size
                changed = True
        elif step_type in {"binary_morphology", "grayscale_morphology"}:
            operations = ["erode", "dilate", "open", "close"]
            if step_type == "grayscale_morphology":
                operations.extend(["white_tophat", "black_tophat"])
            current_operation = params.get("operation", "open")
            operation = st.selectbox(
                "Operation",
                operations,
                index=operations.index(current_operation)
                if current_operation in operations
                else operations.index("open"),
                key=f"{step_id}-operation",
            )
            kernel_size = st.slider(
                "Kernel size",
                min_value=1,
                max_value=31,
                value=int(params.get("kernel_size", 3)),
                step=2,
                key=f"{step_id}-kernel-size",
            )
            iterations = st.number_input(
                "Iterations",
                min_value=1,
                max_value=20,
                value=int(params.get("iterations", 1)),
                step=1,
                key=f"{step_id}-iterations",
            )
            kernel_shape = st.selectbox(
                "Kernel shape",
                ["rect", "ellipse"],
                index=0 if params.get("kernel_shape", "rect") == "rect" else 1,
                format_func=lambda value: (
                    "Rectangular" if value == "rect" else "Elliptical"
                ),
                key=f"{step_id}-kernel-shape",
            )
            kernel_orientation = params.get("kernel_orientation", "square")
            if kernel_shape == "rect":
                orientations = ["square", "horizontal", "vertical"]
                kernel_orientation = st.selectbox(
                    "Rectangle orientation",
                    orientations,
                    index=orientations.index(kernel_orientation)
                    if kernel_orientation in orientations
                    else 0,
                    format_func=lambda value: value.title(),
                    key=f"{step_id}-kernel-orientation",
                )
            new_params = {
                "operation": operation,
                "kernel_size": int(kernel_size),
                "iterations": int(iterations),
                "kernel_shape": kernel_shape,
                "kernel_orientation": kernel_orientation,
            }
            if any(params.get(key) != value for key, value in new_params.items()):
                params.update(new_params)
                changed = True
        else:
            st.error(f"Unsupported pipeline step: {step_type}")

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


def main() -> None:
    st.header("2. Preprocess images")

    extracted_images = st.session_state.get("images", [])
    extracted_pages = st.session_state.get("pages", [])
    with st.sidebar:
        language = st.radio(
            "Document language",
            ["vie", "sino-nom"],
            horizontal=True,
            key="document_language",
        )

        source_options = ["Upload an image"] + [
            f"Extracted page {page}" for page in extracted_pages
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
            source_image = extracted_images[source_index - 1]

        if "preprocess_steps" not in st.session_state:
            st.session_state.preprocess_steps = []
        pipeline = PreprocessPipeline(st.session_state.preprocess_steps)

        st.subheader("Transformation pipeline")
        add_type = st.selectbox(
            "Transformation to add",
            [
                "convert_color",
                "reverse_color",
                "filter",
                "binary_morphology",
                "grayscale_morphology",
            ],
            format_func=_step_label,
            key="preprocess_add_type",
        )
        add_col, reset_col = st.columns(2)
        with add_col:
            if st.button("Add transformation"):
                pipeline.add_step(add_type)
                st.session_state.preprocess_steps = pipeline.to_dict()
                st.rerun()
        with reset_col:
            if st.button("Reset pipeline"):
                pipeline = PreprocessPipeline()
                st.session_state.preprocess_steps = []
                for version in range(1, 5):
                    _set_comparison_text(version, "")
                st.session_state.cmp_v1_annotated = None
                st.session_state.cmp_v3_annotated = None
                st.session_state.cmp_v2_response = None
                st.session_state.cmp_v4_response = None
                st.rerun()

        for index in range(len(pipeline.steps)):
            if _render_step_controls(pipeline, index):
                st.session_state.preprocess_steps = pipeline.to_dict()
                st.rerun()
        st.session_state.preprocess_steps = pipeline.to_dict()
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
                imported_pipeline = _pipeline_from_settings(
                    json.loads(imported_settings.getvalue().decode("utf-8"))
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                st.error(f"Could not import settings: {error}")
            else:
                st.session_state.preprocess_steps = imported_pipeline.to_dict()
                for version in range(1, 5):
                    _set_comparison_text(version, "")
                st.session_state.cmp_v1_annotated = None
                st.session_state.cmp_v3_annotated = None
                st.session_state.cmp_v2_response = None
                st.session_state.cmp_v4_response = None
                st.rerun()

        st.divider()
        min_box_width, min_box_height = _bounding_box_filter_controls()

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
            st.caption(
                f"Bounding-box filter: min width {min_box_width}px, min height {min_box_height}px"
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
    run_all = btn_all.button("⚡ Run 4-Way Comparison", type="primary")
    run_tess = btn_tess.button("Run Tesseract extractions (V1 & V3)")
    run_vlm = btn_vlm.button("Run VLM extractions (V2 & V4)")

    if run_all or run_tess:
        with st.spinner("Running Tesseract OCR on original & preprocessed images…"):
            try:
                tess_ocr_v1 = TesseractOCR(
                    lang="vie",
                    confidence_threshold=LOW_CONFIDENCE_THRESHOLD,
                )
                tess_ocr_v1.run(
                    source_image,
                    config=cmp_psm,
                    min_width=min_box_width,
                    min_height=min_box_height,
                )
                _set_comparison_text(1, tess_ocr_v1.text)
                st.session_state.cmp_v1_annotated = tess_ocr_v1.annotate_image()
                st.session_state.cmp_v1_status = "Success"
            except Exception as error:
                _set_comparison_text(1, f"Error: {error}")
                st.session_state.cmp_v1_annotated = None
                st.session_state.cmp_v1_status = f"Failed ({error})"
                st.error(f"Tesseract (Original) failed: {error}")

            try:
                tess_ocr_v3 = TesseractOCR(
                    lang="vie",
                    confidence_threshold=LOW_CONFIDENCE_THRESHOLD,
                )
                tess_ocr_v3.run(
                    processed_image,
                    config=cmp_psm,
                    min_width=min_box_width,
                    min_height=min_box_height,
                )
                _set_comparison_text(3, tess_ocr_v3.text)
                st.session_state.cmp_v3_annotated = tess_ocr_v3.annotate_image()
                st.session_state.cmp_v3_status = "Success"
            except Exception as error:
                _set_comparison_text(3, f"Error: {error}")
                st.session_state.cmp_v3_annotated = None
                st.session_state.cmp_v3_status = f"Failed ({error})"
                st.error(f"Tesseract (Preprocessed) failed: {error}")

    if run_all or run_vlm:
        with st.spinner("Running VLM OCR on original & preprocessed images…"):
            try:
                vlm_ocr_v2 = VLMOCR(
                    model=vlm_model,
                    options={"temperature": vlm_temperature},
                )
                vlm_ocr_v2.run(source_image)
                _set_comparison_text(2, vlm_ocr_v2.text)
                st.session_state.cmp_v2_response = vlm_ocr_v2.raw_response
                st.session_state.cmp_v2_status = "Success"
            except Exception as error:
                _set_comparison_text(2, f"Error: {error}")
                st.session_state.cmp_v2_response = None
                st.session_state.cmp_v2_status = f"Failed ({error})"
                st.warning(f"VLM (Original) failed: {error}")

            try:
                vlm_ocr_v4 = VLMOCR(
                    model=vlm_model,
                    options={"temperature": vlm_temperature},
                )
                vlm_ocr_v4.run(processed_image)
                _set_comparison_text(4, vlm_ocr_v4.text)
                st.session_state.cmp_v4_response = vlm_ocr_v4.raw_response
                st.session_state.cmp_v4_status = "Success"
            except Exception as error:
                _set_comparison_text(4, f"Error: {error}")
                st.session_state.cmp_v4_response = None
                st.session_state.cmp_v4_status = f"Failed ({error})"
                st.warning(f"VLM (Preprocessed) failed: {error}")

    v1_text = st.session_state.get("cmp_v1_text", "")
    v2_text = st.session_state.get("cmp_v2_text", "")
    v3_text = st.session_state.get("cmp_v3_text", "")
    v4_text = st.session_state.get("cmp_v4_text", "")
    v1_annotated = st.session_state.get("cmp_v1_annotated")
    v3_annotated = st.session_state.get("cmp_v3_annotated")

    has_extractions = any([v1_text, v2_text, v3_text, v4_text])
    if has_extractions:
        tab_side, tab_cols, tab_summary = st.tabs(
            ["Side-by-Side View", "4-Way Text View", "Metrics Summary"]
        )

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

        with tab_cols:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown("**V1: Orig + Tesseract**")
                s1 = _text_stats(v1_text)
                st.caption(f"{s1['chars']}c / {s1['words']}w")
                st.text_area("V1 Text", value=v1_text, height=350, key="v1_col_area")
                st.download_button(
                    "Download V1",
                    data=v1_text,
                    file_name="v1_original_tesseract.txt",
                    disabled=not bool(v1_text),
                    key="dl_v1",
                )
            with c2:
                st.markdown("**V2: Orig + VLM**")
                s2 = _text_stats(v2_text)
                st.caption(f"{s2['chars']}c / {s2['words']}w")
                st.text_area("V2 Text", value=v2_text, height=350, key="v2_col_area")
                st.download_button(
                    "Download V2",
                    data=v2_text,
                    file_name="v2_original_vlm.txt",
                    disabled=not bool(v2_text),
                    key="dl_v2",
                )
            with c3:
                st.markdown("**V3: Prep + Tesseract**")
                s3 = _text_stats(v3_text)
                st.caption(f"{s3['chars']}c / {s3['words']}w")
                st.text_area("V3 Text", value=v3_text, height=350, key="v3_col_area")
                st.download_button(
                    "Download V3",
                    data=v3_text,
                    file_name="v3_preprocessed_tesseract.txt",
                    disabled=not bool(v3_text),
                    key="dl_v3",
                )
            with c4:
                st.markdown("**V4: Prep + VLM**")
                s4 = _text_stats(v4_text)
                st.caption(f"{s4['chars']}c / {s4['words']}w")
                st.text_area("V4 Text", value=v4_text, height=350, key="v4_col_area")
                st.download_button(
                    "Download V4",
                    data=v4_text,
                    file_name="v4_preprocessed_vlm.txt",
                    disabled=not bool(v4_text),
                    key="dl_v4",
                )

        with tab_summary:
            summary_data = [
                {
                    "Version": "V1",
                    "Source Image": "Original",
                    "OCR Engine": "Tesseract",
                    "Characters": _text_stats(v1_text)["chars"],
                    "Words": _text_stats(v1_text)["words"],
                    "Paragraphs": _text_stats(v1_text)["paragraphs"],
                    "Status": st.session_state.get("cmp_v1_status", "Not run"),
                },
                {
                    "Version": "V2",
                    "Source Image": "Original",
                    "OCR Engine": "VLM",
                    "Characters": _text_stats(v2_text)["chars"],
                    "Words": _text_stats(v2_text)["words"],
                    "Paragraphs": _text_stats(v2_text)["paragraphs"],
                    "Status": st.session_state.get("cmp_v2_status", "Not run"),
                },
                {
                    "Version": "V3",
                    "Source Image": "Preprocessed",
                    "OCR Engine": "Tesseract",
                    "Characters": _text_stats(v3_text)["chars"],
                    "Words": _text_stats(v3_text)["words"],
                    "Paragraphs": _text_stats(v3_text)["paragraphs"],
                    "Status": st.session_state.get("cmp_v3_status", "Not run"),
                },
                {
                    "Version": "V4",
                    "Source Image": "Preprocessed",
                    "OCR Engine": "VLM",
                    "Characters": _text_stats(v4_text)["chars"],
                    "Words": _text_stats(v4_text)["words"],
                    "Paragraphs": _text_stats(v4_text)["paragraphs"],
                    "Status": st.session_state.get("cmp_v4_status", "Not run"),
                },
            ]
            st.dataframe(pd.DataFrame(summary_data), width="stretch")
    else:
        st.info(
            "Click '⚡ Run 4-Way Comparison' to run extractions on both images with both engines."
        )

    apply_disabled = not bool(extracted_images)
    if st.button(
        "Apply pipeline to all extracted images and open OCR",
        type="secondary",
        disabled=apply_disabled,
        help="Extract images from the first page before using this action.",
    ):
        with st.spinner("Applying preprocessing to extracted images…"):
            try:
                processed_images = [pipeline.apply(image) for image in extracted_images]
            except (TypeError, ValueError, cv2.error) as error:
                st.error(f"Could not process all extracted images: {error}")
            else:
                st.session_state.ocr_images = processed_images
                st.session_state.ocr_pages = list(extracted_pages)
                st.session_state.selected_image = 0
                st.session_state.annotated_image = None
                st.session_state.annotated_for = None
                st.session_state.annotated_config = None
                st.switch_page("pages/3_ocr_image.py")


if __name__ == "__main__":
    main()
