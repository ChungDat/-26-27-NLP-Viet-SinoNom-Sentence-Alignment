from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

try:
    from .extraction import pdf_to_images, select_pages
    from .ocr_viet import get_ocr_data, normalize_text
    from .preprocess import PreprocessPipeline
except ImportError:
    from extraction import pdf_to_images, select_pages
    from ocr_viet import get_ocr_data, normalize_text
    from preprocess import PreprocessPipeline


OUTPUT_DIR = Path("Dong_hang")
LOW_CONFIDENCE_THRESHOLD = 80.0
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
    """Return the number of pages in an uploaded PDF.
    """

    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        return len(document)
    finally:
        document.close()


def _image_to_pil(image: np.ndarray) -> Image.Image:
    """Convert an RGB or grayscale NumPy image to a PIL image."""

    if image.ndim == 2:
        return Image.fromarray(image)
    if image.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGBA2RGB))
    return Image.fromarray(image)


def filter_bounding_boxes(
    data: dict,
    min_width: int = 0,
    min_height: int = 0,
) -> dict:
    """Return OCR data containing only boxes that meet both size limits.

    A box is rejected when either its width or height is below the requested
    minimum. Every list-valued OCR field is filtered by the same indices, so
    its values remain aligned with the retained ``text`` entries.
    """

    if min_width < 0 or min_height < 0:
        raise ValueError("minimum bounding-box dimensions must be non-negative")

    text_entries = data.get("text", [])
    widths = data.get("width", [])
    heights = data.get("height", [])
    retained_indices: list[int] = []
    for index in range(len(text_entries)):
        try:
            width = int(widths[index])
            height = int(heights[index])
        except (IndexError, TypeError, ValueError):
            continue
        if width >= min_width and height >= min_height:
            retained_indices.append(index)

    filtered_data: dict = {}
    for field, values in data.items():
        if isinstance(values, list):
            filtered_data[field] = [
                values[index]
                for index in retained_indices
                if index < len(values)
            ]
        else:
            filtered_data[field] = values
    return filtered_data


def _filtered_ocr_text(data: dict) -> str:
    """Rebuild text from retained OCR word boxes in Tesseract reading order."""

    lines: list[str] = []
    words: list[str] = []
    current_line: tuple[int, int, int] | None = None
    current_paragraph: tuple[int, int] | None = None
    previous_paragraph: tuple[int, int] | None = None

    def value(field: str, index: int, default: int = 0) -> int:
        try:
            return int(data.get(field, [])[index])
        except (IndexError, TypeError, ValueError):
            return default

    def confidence(index: int) -> float:
        try:
            return float(data.get("conf", [])[index])
        except (IndexError, TypeError, ValueError):
            return -1.0

    def finish_line() -> None:
        nonlocal words, previous_paragraph
        if not words:
            return
        if previous_paragraph is not None and current_paragraph != previous_paragraph:
            lines.append("")
        lines.append(" ".join(words))
        previous_paragraph = current_paragraph
        words = []

    for index, raw_text in enumerate(data.get("text", [])):
        text = str(raw_text).strip()
        if not text or confidence(index) < 0:
            continue

        paragraph = (value("block_num", index), value("par_num", index))
        line = (*paragraph, value("line_num", index))
        if current_line is not None and line != current_line:
            finish_line()
        current_line = line
        current_paragraph = paragraph
        words.append(text)

    finish_line()
    return normalize_text("\n".join(lines))


def _annotated_image(image: np.ndarray, data: dict) -> Image.Image:
    """Draw OCR word boxes, marking low-confidence detections in red."""

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
        color = (255, 0, 0) if confidence < LOW_CONFIDENCE_THRESHOLD else (0, 191, 255)

        draw.rectangle(
            (left, top, left + width, top + height),
            outline=color,
            width=3,
        )

        draw.text((left, max(top - 50, 0)), label, font=font, fill=color)

    return result


def _ocr_image(
    image: np.ndarray,
    config: str,
    min_width: int = 10,
    min_height: int = 10,
) -> tuple[str, Image.Image]:
    """
    Run OCR on an in-memory image and return text plus an annotated image
    """

    image_for_encoding = image
    if image.ndim == 3 and image.shape[2] == 3:
        image_for_encoding = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image_for_encoding = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    with tempfile.NamedTemporaryFile(suffix=".jpg") as image_file:
        success, encoded = cv2.imencode(".jpg", image_for_encoding)
        if not success:
            raise ValueError("Could not encode the selected image")
        image_file.write(encoded.tobytes())
        image_file.flush()
        data = get_ocr_data(image_file.name, config=config)
    filtered_data = filter_bounding_boxes(data, min_width, min_height)
    return _filtered_ocr_text(filtered_data), _annotated_image(image, filtered_data)


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
        end = st.number_input("End page (Included)", min_value=1, max_value=page_count, value=int(start))
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
                    images = pdf_to_images(pdf_file.name, pages, dpi=300)
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
        st.success(f"Extracted {len(images)} image(s). Open the OCR page to continue.")


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


def _ocr_page() -> None:
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
                    text, annotated = _ocr_image(
                        current_image,
                        psm_config,
                        min_box_width,
                        min_box_height,
                    )
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
                format_func=lambda value: "Rectangular" if value == "rect" else "Elliptical",
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


def _preprocess_page() -> None:
    st.header("2. Preprocess images")

    extracted_images = st.session_state.get("images", [])
    extracted_pages = st.session_state.get("pages", [])
    with st.sidebar:
        st.divider()
        source_options = ["Upload an image"] + [
            f"Extracted page {page}"
            for page in extracted_pages
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
                st.session_state.preprocess_ocr_text = ""
                st.session_state.preprocess_ocr_text_editor = ""
                st.session_state.preprocess_annotated = None
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
            accept_multiple_files=False
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
                st.session_state.preprocess_ocr_text = ""
                st.session_state.preprocess_ocr_text_editor = ""
                st.session_state.preprocess_annotated = None
                st.session_state.preprocess_annotated_filter = None
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

    st.subheader("Try OCR on the preview")
    preview_psm = st.selectbox(
        "Page segmentation method",
        list(PSM_NOTES),
        format_func=lambda config: f"{config} - {PSM_NOTES[config]}",
        key="preprocess_psm_config",
    )
    if st.button("Run OCR on processed preview", type="primary"):
        with st.spinner("Running OCR…"):
            try:
                text, annotated = _ocr_image(
                    processed_image,
                    preview_psm,
                    min_box_width,
                    min_box_height,
                )
            except Exception as error:
                st.error(f"OCR failed: {error}")
            else:
                st.session_state.preprocess_ocr_text = text
                st.session_state.preprocess_ocr_text_editor = text
                st.session_state.preprocess_annotated = annotated
                st.session_state.preprocess_annotated_filter = (
                    preview_psm,
                    min_box_width,
                    min_box_height,
                )

    if (
        st.session_state.get("preprocess_annotated") is not None
        and st.session_state.get("preprocess_annotated_filter")
        == (preview_psm, min_box_width, min_box_height)
    ):
        st.image(
            st.session_state.preprocess_annotated,
            caption="OCR preview",
            width="stretch",
        )
        st.text_area(
            "OCR extraction text",
            height=260,
            key="preprocess_ocr_text_editor",
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
                st.session_state.open_ocr_after_apply = True
                st.rerun()


def main() -> None:
    st.set_page_config(page_title="PDF OCR", layout="wide")
    if st.session_state.pop("open_ocr_after_apply", False):
        # This executes before the radio is created on this rerun.
        st.session_state.app_page = "Run OCR"
    st.title("PDF OCR workspace")
    page = st.sidebar.radio(
        "Page",
        ["Extract images", "Preprocess", "Run OCR"],
        key="app_page",
    )
    if page == "Extract images":
        _load_page()
    elif page == "Preprocess":
        _preprocess_page()
    else:
        _ocr_page()


if __name__ == "__main__":
    main()
