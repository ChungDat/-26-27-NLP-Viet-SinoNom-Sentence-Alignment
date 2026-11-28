from __future__ import annotations

import concurrent.futures
import hashlib
import html
import json
import re
import threading
import time
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

try:
    import Levenshtein
except ImportError:
    Levenshtein = None  # type: ignore[assignment]

try:
    from ..ocr_viet import (
        VLMOCR,
        OCRPage,
        OCRParagraph,
        TesseractOCR,
    )
except ImportError:
    from ocr_viet import (  # type: ignore[no-redef]
        VLMOCR,
        OCRPage,
        OCRParagraph,
        TesseractOCR,
    )

try:
    from ..ppocr import (
        PPOCRLabelPage,
        make_image_path,
        normalize_transcription,
        ocr_page_to_ppocr,
        parse_header_paragraph,
        write_file_state_txt,
        write_label_txt,
    )
except ImportError:
    from ppocr import (  # type: ignore[no-redef]
        PPOCRLabelPage,
        make_image_path,
        normalize_transcription,
        ocr_page_to_ppocr,
        parse_header_paragraph,
        write_file_state_txt,
        write_label_txt,
    )

try:
    from ..preprocess import PreprocessPipeline
except ImportError:
    from preprocess import PreprocessPipeline  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOW_CONFIDENCE_THRESHOLD = 85.0
VLM_MODEL = "qwen3-vl:2b-instruct"
FUZZY_MATCH_THRESHOLD = 0.75

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

# ---------------------------------------------------------------------------
# Global thread-safe background OCR prefetch cache
# ---------------------------------------------------------------------------

try:
    from ..ocr_cache import PREFETCH_CACHE, PREFETCH_LOCK, PREFETCH_IN_PROGRESS
except ImportError:
    from ocr_cache import PREFETCH_CACHE, PREFETCH_LOCK, PREFETCH_IN_PROGRESS

# Aliases pointing to persistent singleton objects in sys.modules['src.ocr_cache']
_PREFETCH_CACHE = PREFETCH_CACHE
_PREFETCH_LOCK = PREFETCH_LOCK
_PREFETCH_IN_PROGRESS = PREFETCH_IN_PROGRESS


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------


def _image_to_pil(image: np.ndarray) -> Image.Image:
    """Convert an RGB or grayscale NumPy image to a PIL image.

    Args:
        image: Input NumPy array (2-D grayscale, 3-channel RGB, or 4-channel RGBA).

    Returns:
        A PIL Image.
    """
    if image.ndim == 2:
        return Image.fromarray(image)
    if image.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGBA2RGB))
    return Image.fromarray(image)


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on blank lines; strip empty results and normalize text.

    Args:
        text: Multi-paragraph text with blank-line paragraph separators.

    Returns:
        List of non-empty paragraph strings with NFC and punctuation normalization.
    """
    raw_paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [normalize_transcription(p) for p in raw_paras if normalize_transcription(p)]


def _draw_paragraph_bboxes(
    image: np.ndarray | Image.Image,
    paragraphs: list[OCRParagraph],
    box_width: int = 2,
    color: tuple[int, int, int] = (0, 255, 0),
    font_size: int = 20,
) -> Image.Image:
    """Draw paragraph-level bounding boxes and labels on an image."""
    from PIL import ImageDraw, ImageFont

    pil_image = _image_to_pil(image).copy()
    draw = ImageDraw.Draw(pil_image)

    try:
        font_path = Path(__file__).parent.parent / "fonts" / "NotoSans-Regular.ttf"
        font = ImageFont.truetype(str(font_path), font_size)
    except OSError:
        font = ImageFont.load_default(font_size)

    for p in paragraphs:
        if p.bbox is not None:
            x0, y0, x1, y1 = p.bbox
            draw.rectangle((x0, y0, x1, y1), outline=color, width=box_width)
            draw.text(
                (x0, max(y0 - font_size - 4, 0)),
                f"P{p.paragraph_id}",
                font=font,
                fill=color,
            )

    return pil_image


def _compute_default_paragraph_bboxes(
    reviewed_paragraphs: list[str],
    tess_page: OCRPage | None = None,
    filtered_raw: dict | None = None,
) -> list[tuple[int, int, int, int] | None]:
    """Calculate default bounding boxes for reviewed paragraphs."""
    if not reviewed_paragraphs:
        return []

    # Case 1: Paragraph counts match
    if (
        tess_page is not None
        and tess_page.paragraphs
        and len(reviewed_paragraphs) == len(tess_page.paragraphs)
    ):
        return [p.bbox for p in tess_page.paragraphs]

    # Case 2: Paragraph counts differ — compute paragraph bounding box from word bboxes
    word_boxes: list[tuple[str, tuple[int, int, int, int]]] = []
    if filtered_raw:
        raw_texts = filtered_raw.get("text", [])
        raw_lefts = filtered_raw.get("left", [])
        raw_tops = filtered_raw.get("top", [])
        raw_widths = filtered_raw.get("width", [])
        raw_heights = filtered_raw.get("height", [])
        for i in range(len(raw_texts)):
            w_text = str(raw_texts[i]).strip()
            if not w_text:
                continue
            try:
                l, t, w, h = (
                    int(raw_lefts[i]),
                    int(raw_tops[i]),
                    int(raw_widths[i]),
                    int(raw_heights[i]),
                )
                if w > 0 and h > 0:
                    word_boxes.append((w_text, (l, t, l + w, t + h)))
            except (IndexError, TypeError, ValueError):
                continue

    word_idx = 0
    total_words = len(word_boxes)
    bboxes: list[tuple[int, int, int, int] | None] = []

    for idx, para_text in enumerate(reviewed_paragraphs, start=1):
        para_words = para_text.split()
        matched_boxes: list[tuple[int, int, int, int]] = []

        start_w = word_idx
        for pw in para_words:
            found = False
            for look in range(word_idx, min(word_idx + 10, total_words)):
                if (
                    word_boxes[look][0] == pw
                    or word_boxes[look][0] in pw
                    or pw in word_boxes[look][0]
                ):
                    matched_boxes.append(word_boxes[look][1])
                    word_idx = look + 1
                    found = True
                    break

        if matched_boxes:
            x_min = min(b[0] for b in matched_boxes)
            y_min = min(b[1] for b in matched_boxes)
            x_max = max(b[2] for b in matched_boxes)
            y_max = max(b[3] for b in matched_boxes)
            bboxes.append((x_min, y_min, x_max, y_max))
        elif (
            tess_page and tess_page.paragraphs and (idx - 1) < len(tess_page.paragraphs)
        ):
            bboxes.append(tess_page.paragraphs[idx - 1].bbox)
        else:
            bboxes.append(None)

    return bboxes


def _sync_paragraph_bboxes(
    current_bboxes: list[dict],
    reviewed_paragraphs: list[str],
    tess_page: OCRPage | None = None,
    filtered_raw: dict | None = None,
) -> list[dict]:
    """Synchronize paragraph bounding box rows with reviewed paragraphs."""
    computed_defaults = _compute_default_paragraph_bboxes(
        reviewed_paragraphs, tess_page=tess_page, filtered_raw=filtered_raw
    )
    synced: list[dict] = []

    for i, para_text in enumerate(reviewed_paragraphs):
        preview = para_text[:40] + ("..." if len(para_text) > 40 else "")
        para_id = i + 1

        # Reuse existing values if present
        if i < len(current_bboxes):
            existing = current_bboxes[i]
            synced.append(
                {
                    "Paragraph": f"P{para_id}",
                    "x_min": existing.get("x_min"),
                    "y_min": existing.get("y_min"),
                    "x_max": existing.get("x_max"),
                    "y_max": existing.get("y_max"),
                    "Preview": preview,
                }
            )
        else:
            # New paragraph from line break split
            def_box = computed_defaults[i] if i < len(computed_defaults) else None
            synced.append(
                {
                    "Paragraph": f"P{para_id}",
                    "x_min": def_box[0] if def_box else 0,
                    "y_min": def_box[1] if def_box else 0,
                    "x_max": def_box[2] if def_box else 0,
                    "y_max": def_box[3] if def_box else 0,
                    "Preview": preview,
                }
            )

    return synced


def _clamp_bbox_pair(low: object, high: object, max_bound: int) -> tuple[int, int]:
    """Clamp and order a (min, max) coordinate pair within [0, max_bound]."""
    try:
        v_low = int(low) if low is not None else 0
    except (TypeError, ValueError):
        v_low = 0
    try:
        v_high = int(high) if high is not None else 0
    except (TypeError, ValueError):
        v_high = 0
    v_low = max(0, min(v_low, max_bound))
    v_high = max(0, min(v_high, max_bound))
    if v_low > v_high:
        v_low, v_high = v_high, v_low
    return v_low, v_high


def _on_bbox_slider_change(
    bbox_key: str,
    para_idx: int,
    slider_key: str,
    min_key: str,
    max_key: str,
    min_field: str,
    max_field: str,
) -> None:
    """Sync number inputs and bbox session state when a range slider changes."""
    v_low, v_high = st.session_state[slider_key]
    st.session_state[min_key] = int(v_low)
    st.session_state[max_key] = int(v_high)
    rows = st.session_state.get(bbox_key, [])
    if para_idx < len(rows):
        rows[para_idx][min_field] = int(v_low)
        rows[para_idx][max_field] = int(v_high)


def _on_bbox_inputs_change(
    bbox_key: str,
    para_idx: int,
    slider_key: str,
    min_key: str,
    max_key: str,
    min_field: str,
    max_field: str,
    max_bound: int,
) -> None:
    """Sync range slider and bbox session state when a numeric input changes."""
    v_low, v_high = _clamp_bbox_pair(
        st.session_state.get(min_key, 0),
        st.session_state.get(max_key, 0),
        max_bound,
    )
    st.session_state[min_key] = v_low
    st.session_state[max_key] = v_high
    st.session_state[slider_key] = (v_low, v_high)
    rows = st.session_state.get(bbox_key, [])
    if para_idx < len(rows):
        rows[para_idx][min_field] = v_low
        rows[para_idx][max_field] = v_high


def _render_paragraph_bbox_controls(
    index: int,
    synced_bbox_rows: list[dict],
    bbox_session_key: str,
    img_width: int,
    img_height: int,
) -> None:
    """Render per-paragraph tabs with X/Y range sliders and numeric inputs."""
    if not synced_bbox_rows:
        st.caption("No paragraphs detected to adjust bounding boxes.")
        return

    max_x = max(1, int(img_width))
    max_y = max(1, int(img_height))
    tab_labels = [
        row.get("Paragraph", f"P{i + 1}") for i, row in enumerate(synced_bbox_rows)
    ]
    tabs = st.tabs(tab_labels)

    axes = (
        ("X range", "x_min", "x_max", max_x, "x"),
        ("Y range", "y_min", "y_max", max_y, "y"),
    )
    for i, (tab, row) in enumerate(zip(tabs, synced_bbox_rows)):
        with tab:
            for label, min_field, max_field, max_bound, axis_tag in axes:
                slider_key = f"ocr_bbox_ctrl_{index}_{i}_{axis_tag}_slider"
                min_key = f"ocr_bbox_ctrl_{index}_{i}_{min_field}"
                max_key = f"ocr_bbox_ctrl_{index}_{i}_{max_field}"

                v_low, v_high = _clamp_bbox_pair(
                    row.get(min_field), row.get(max_field), max_bound
                )
                st.session_state[slider_key] = (v_low, v_high)
                st.session_state[min_key] = v_low
                st.session_state[max_key] = v_high

                st.slider(
                    label,
                    min_value=0,
                    max_value=max_bound,
                    step=1,
                    key=slider_key,
                    on_change=_on_bbox_slider_change,
                    args=(
                        bbox_session_key,
                        i,
                        slider_key,
                        min_key,
                        max_key,
                        min_field,
                        max_field,
                    ),
                )
                col_min, col_max = st.columns(2)
                with col_min:
                    st.number_input(
                        min_field,
                        min_value=0,
                        max_value=max_bound,
                        step=1,
                        key=min_key,
                        on_change=_on_bbox_inputs_change,
                        args=(
                            bbox_session_key,
                            i,
                            slider_key,
                            min_key,
                            max_key,
                            min_field,
                            max_field,
                            max_bound,
                        ),
                    )
                with col_max:
                    st.number_input(
                        max_field,
                        min_value=0,
                        max_value=max_bound,
                        step=1,
                        key=max_key,
                        on_change=_on_bbox_inputs_change,
                        args=(
                            bbox_session_key,
                            i,
                            slider_key,
                            min_key,
                            max_key,
                            min_field,
                            max_field,
                            max_bound,
                        ),
                    )


def _build_ocr_page(
    text: str,
    pdf_page_number: int,
    document_page_number: int,
    tess_page: OCRPage | None = None,
    filtered_raw: dict | None = None,
    custom_bboxes: list[tuple[int, int, int, int] | None] | None = None,
) -> OCRPage:
    """Build a reviewed OCRPage from plain editor text.

    Assigns bounding boxes to paragraphs by:
    1. If the reviewed paragraph count matches the Tesseract paragraph count,
       directly reusing each Tesseract paragraph's bounding box.
    2. Otherwise, using Tesseract word/line bounding boxes from ``filtered_raw``
       to find matching words for the reviewed paragraph text and unioning their
       bounding boxes.
    3. Falling back to the closest Tesseract paragraph bbox, or None.

    Args:
        text: Editor text with blank-line paragraph separators.
        pdf_page_number: 1-indexed PDF reader page number.
        document_page_number: Actual document page number after offset.
        tess_page: Optional Tesseract OCRPage result for this page.
        filtered_raw: Optional Tesseract raw output dict (with left, top, width, height, text).

    Returns:
        An OCRPage with method="reviewed".
    """
    reviewed_paragraphs = _split_paragraphs(text)
    paragraphs: list[OCRParagraph] = []

    if custom_bboxes is not None and len(custom_bboxes) == len(reviewed_paragraphs):
        for idx, (para_text, bbox) in enumerate(
            zip(reviewed_paragraphs, custom_bboxes), start=1
        ):
            paragraphs.append(
                OCRParagraph(
                    paragraph_id=idx,
                    bbox=bbox,
                    text=para_text,
                )
            )
        return OCRPage(
            pdf_page_number=pdf_page_number,
            document_page_number=document_page_number,
            page_name=f"{pdf_page_number:03d}",
            method="reviewed",
            paragraphs=paragraphs,
        )

    # Case 1: Paragraph counts match
    if (
        tess_page is not None
        and tess_page.paragraphs
        and len(reviewed_paragraphs) == len(tess_page.paragraphs)
    ):
        for idx, para_text in enumerate(reviewed_paragraphs, start=1):
            tess_p = tess_page.paragraphs[idx - 1]
            paragraphs.append(
                OCRParagraph(
                    paragraph_id=idx,
                    bbox=tess_p.bbox,
                    text=para_text,
                )
            )
        return OCRPage(
            pdf_page_number=pdf_page_number,
            document_page_number=document_page_number,
            page_name=f"{pdf_page_number:03d}",
            method="reviewed",
            paragraphs=paragraphs,
        )

    # Case 2: Paragraph counts differ — compute paragraph bounding box from word bboxes
    word_boxes: list[tuple[str, tuple[int, int, int, int]]] = []
    if filtered_raw:
        raw_texts = filtered_raw.get("text", [])
        raw_lefts = filtered_raw.get("left", [])
        raw_tops = filtered_raw.get("top", [])
        raw_widths = filtered_raw.get("width", [])
        raw_heights = filtered_raw.get("height", [])
        for i in range(len(raw_texts)):
            w_text = str(raw_texts[i]).strip()
            if not w_text:
                continue
            try:
                l, t, w, h = (
                    int(raw_lefts[i]),
                    int(raw_tops[i]),
                    int(raw_widths[i]),
                    int(raw_heights[i]),
                )
                if w > 0 and h > 0:
                    word_boxes.append((w_text, (l, t, l + w, t + h)))
            except (IndexError, TypeError, ValueError):
                continue

    word_idx = 0
    total_words = len(word_boxes)

    for idx, para_text in enumerate(reviewed_paragraphs, start=1):
        para_words = para_text.split()
        matched_boxes: list[tuple[int, int, int, int]] = []

        # Consume matching or consecutive words in reading order
        start_w = word_idx
        for pw in para_words:
            # Look ahead a small window for the word
            found = False
            for look in range(word_idx, min(word_idx + 10, total_words)):
                if (
                    word_boxes[look][0] == pw
                    or word_boxes[look][0] in pw
                    or pw in word_boxes[look][0]
                ):
                    matched_boxes.append(word_boxes[look][1])
                    word_idx = look + 1
                    found = True
                    break
            if not found and word_idx < total_words:
                # If exact word not found, still include current position's bbox if within reasonable count
                pass

        if matched_boxes:
            x_min = min(b[0] for b in matched_boxes)
            y_min = min(b[1] for b in matched_boxes)
            x_max = max(b[2] for b in matched_boxes)
            y_max = max(b[3] for b in matched_boxes)
            p_bbox = (x_min, y_min, x_max, y_max)
        elif (
            tess_page and tess_page.paragraphs and (idx - 1) < len(tess_page.paragraphs)
        ):
            p_bbox = tess_page.paragraphs[idx - 1].bbox
        else:
            p_bbox = None

        paragraphs.append(
            OCRParagraph(
                paragraph_id=idx,
                bbox=p_bbox,
                text=para_text,
            )
        )

    return OCRPage(
        pdf_page_number=pdf_page_number,
        document_page_number=document_page_number,
        page_name=f"{pdf_page_number:03d}",
        method="reviewed",
        paragraphs=paragraphs,
    )


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------


def _clear_ocr_session_state() -> None:
    """Clear OCR page session state keys, preserving confirmed review work.

    Clears cache keys and prefetch state without touching confirmed pages or
    flags, so in-progress review is preserved when only OCR config changes.
    """
    _OCR_CACHE_KEYS = [
        "ocr_current_index",
        "ocr_page_cache",
        "ocr_images_signature",
        "ocr_preprocess_signature",
    ]
    for key in _OCR_CACHE_KEYS:
        st.session_state.pop(key, None)
    for key in list(st.session_state.keys()):
        if key.startswith(
            (
                "ocr_editor_widget_",
                "ocr_vlm_widget_",
                "image_view_mode_",
                "ocr_bbox_list_",
                "ocr_bbox_ctrl_",
            )
        ):
            del st.session_state[key]
    with _PREFETCH_LOCK:
        _PREFETCH_CACHE.clear()
        _PREFETCH_IN_PROGRESS.clear()


def _clear_all_ocr_session_state() -> None:
    """Clear ALL OCR page state including confirmed pages (used when new PDF extracted).

    Should be called whenever the source PDF or extracted images change, to
    ensure a clean slate for a new document.
    """
    _clear_ocr_session_state()
    for key in [
        "ocr_confirmed_flags",
        "ocr_confirmed_pages",
        "page_adjustment",
        "save_alert",
        "show_inline_overwrite_warning",
        "overwrite_approved",
    ]:
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Warning computation & alignment
# ---------------------------------------------------------------------------


def _fuzzy_ratio(a: str, b: str) -> float:
    """Return Levenshtein similarity ratio between two strings in [0.0, 1.0].

    Uses python-Levenshtein when available, otherwise calculates exact Levenshtein
    ratio using dynamic programming (substitutions cost 2, indels cost 1).

    Args:
        a: First string.
        b: Second string.

    Returns:
        Float in [0.0, 1.0] where 1.0 means identical.
    """
    if Levenshtein is not None:
        return Levenshtein.ratio(a, b)
    if a == b:
        return 1.0
    len_a, len_b = len(a), len(b)
    total_len = len_a + len_b
    if total_len == 0:
        return 1.0
    dp = list(range(len_b + 1))
    for i, ca in enumerate(a):
        new_dp = [i + 1] * (len_b + 1)
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 2
            new_dp[j + 1] = min(dp[j + 1] + 1, new_dp[j] + 1, dp[j] + cost)
        dp = new_dp
    return max(0.0, (total_len - dp[len_b]) / total_len)


def _align_paragraph_words(
    editor_words: list[str],
    vlm_words: list[str],
    max_lookahead: int = 5,
) -> tuple[list[dict], set[int], set[int]]:
    """Realign word sequences using 2-pointer with bounded lookahead.

    Uses Levenshtein fuzzy matching to distinguish soft mismatches (close
    spelling) from hard mismatches (very different words).

    Args:
        editor_words: Words from the editor text in order.
        vlm_words: Words from the VLM reference text in order.
        max_lookahead: Maximum number of words to look ahead for an anchor match.

    Returns:
        A 3-tuple of:
          - discrepancies: List of dicts describing each mismatch.
          - hard_mismatch_indices: Set of editor word indices with hard mismatches.
          - soft_mismatch_indices: Set of editor word indices with soft (fuzzy) mismatches.
    """
    discrepancies: list[dict] = []
    hard_mismatch_indices: set[int] = set()
    soft_mismatch_indices: set[int] = set()
    missing_insertions: dict[int, list[str]] = defaultdict(list)

    i = 0
    j = 0
    m = len(editor_words)
    n = len(vlm_words)

    while i < m and j < n:
        if editor_words[i] == vlm_words[j]:
            i += 1
            j += 1
            continue

        # Check if this is a fuzzy (soft) match first
        ratio = _fuzzy_ratio(editor_words[i], vlm_words[j])
        if ratio >= FUZZY_MATCH_THRESHOLD and editor_words[i] != vlm_words[j]:
            soft_mismatch_indices.add(i)
            discrepancies.append(
                {
                    "type": "soft_mismatch",
                    "editor_segment": editor_words[i],
                    "vlm_segment": vlm_words[j],
                    "msg": f'Fuzzy match: "{editor_words[i]}" ≈ "{vlm_words[j]}" (similarity {ratio:.0%})',
                }
            )
            i += 1
            j += 1
            continue

        # Search for the nearest matching anchor within lookahead window
        best_anchor: tuple[int, int] | None = None
        for total_dist in range(1, 2 * max_lookahead + 1):
            for di in range(min(total_dist, max_lookahead) + 1):
                dj = total_dist - di
                if dj <= max_lookahead and (i + di) < m and (j + dj) < n:
                    # Use fuzzy ratio as the matching condition for anchors
                    if (
                        _fuzzy_ratio(editor_words[i + di], vlm_words[j + dj])
                        >= FUZZY_MATCH_THRESHOLD
                    ):
                        best_anchor = (di, dj)
                        break
            if best_anchor is not None:
                break

        if best_anchor is not None:
            di, dj = best_anchor
            skipped_a = editor_words[i : i + di]
            skipped_b = vlm_words[j : j + dj]

            seg_a = " ".join(skipped_a)
            seg_b = " ".join(skipped_b)

            if di == 0 and dj > 0:
                msg = f'Missing word(s) in editor (VLM has: "{seg_b}")'
                missing_insertions[i].append(seg_b)
            elif di > 0 and dj == 0:
                msg = f'Extra word(s) in editor: "{seg_a}" (not in VLM)'
                for idx in range(i, i + di):
                    hard_mismatch_indices.add(idx)
            else:
                msg = f'Current: "{seg_a}" | VLM: "{seg_b}"'
                for idx in range(i, i + di):
                    hard_mismatch_indices.add(idx)

            discrepancies.append(
                {
                    "type": "mismatch",
                    "editor_segment": seg_a,
                    "vlm_segment": seg_b,
                    "msg": msg,
                }
            )
            i += di
            j += dj
        else:
            # Fallback: advance both pointers by 1 — hard mismatch
            seg_a = editor_words[i]
            seg_b = vlm_words[j]
            hard_mismatch_indices.add(i)
            discrepancies.append(
                {
                    "type": "mismatch",
                    "editor_segment": seg_a,
                    "vlm_segment": seg_b,
                    "msg": f'Current: "{seg_a}" | VLM: "{seg_b}"',
                }
            )
            i += 1
            j += 1

    # Trailing words in editor — hard mismatch
    if i < m:
        for idx in range(i, m):
            hard_mismatch_indices.add(idx)
        seg_a = " ".join(editor_words[i:])
        discrepancies.append(
            {
                "type": "mismatch",
                "editor_segment": seg_a,
                "vlm_segment": "",
                "msg": f'Extra word(s) in editor: "{seg_a}" (not in VLM)',
            }
        )
    # Trailing words in VLM — missing insertion
    elif j < n:
        seg_b = " ".join(vlm_words[j:])
        missing_insertions[m].append(seg_b)
        discrepancies.append(
            {
                "type": "mismatch",
                "editor_segment": "",
                "vlm_segment": seg_b,
                "msg": f'Missing word(s) in editor (VLM has: "{seg_b}")',
            }
        )

    return discrepancies, hard_mismatch_indices, soft_mismatch_indices


def _compute_warnings(
    vlm_source: OCRPage | str | None,
    editor_text: str,
    max_lookahead: int = 5,
) -> tuple[
    list[dict],
    dict[int, set[int]],  # hard_mismatch_indices_by_para
    dict[int, set[int]],  # soft_mismatch_indices_by_para
    dict[int, dict[int, list[str]]],  # missing_words_by_para
    dict[int, str],  # missing_paras_by_idx
]:
    """Compute per-paragraph alignment warnings between editor text and VLM source.

    Args:
        vlm_source: VLM OCR result as OCRPage, plain string, or None.
        editor_text: Current editor text with blank-line paragraph separators.
        max_lookahead: Maximum lookahead for anchor search in word alignment.

    Returns:
        A 5-tuple of:
          - warnings: List of warning dicts with keys type, para, label, msg.
          - hard_mismatch_indices_by_para: Maps para index → set of hard-mismatch word indices.
          - soft_mismatch_indices_by_para: Maps para index → set of soft-mismatch word indices.
          - missing_words_by_para: Maps para index → {word_idx: [missing_vlm_strings]}.
          - missing_paras_by_idx: Maps para index → full VLM paragraph text for orphan VLM paras.
    """
    warnings: list[dict] = []
    hard_mismatch_indices_by_para: dict[int, set[int]] = {}
    soft_mismatch_indices_by_para: dict[int, set[int]] = {}
    missing_words_by_para: dict[int, dict[int, list[str]]] = {}
    missing_paras_by_idx: dict[int, str] = {}

    editor_paras = _split_paragraphs(editor_text)

    if isinstance(vlm_source, str):
        vlm_paras_text = _split_paragraphs(vlm_source)
    elif vlm_source is not None:
        vlm_paras_text = [p.text for p in vlm_source.paragraphs]
    else:
        vlm_paras_text = []

    for i, (ep_text, vp_text) in enumerate(zip_longest(editor_paras, vlm_paras_text)):
        para_label = f"Paragraph {i + 1}"

        # Orphan VLM paragraph — missing in editor
        if ep_text is None and vp_text is not None:
            warnings.append(
                {
                    "type": "orphan",
                    "source": "vlm",
                    "para": i,
                    "label": para_label,
                    "msg": f'Missing paragraph in editor (VLM has: "{vp_text[:60]}…")',
                }
            )
            missing_paras_by_idx[i] = vp_text
            continue

        # Orphan editor paragraph — not in VLM
        if vp_text is None and ep_text is not None:
            warnings.append(
                {
                    "type": "orphan",
                    "source": "editor",
                    "para": i,
                    "label": para_label,
                    "msg": f'Extra paragraph in editor (not in VLM): "{ep_text[:60]}…"',
                }
            )
            # All words in orphan editor paragraph are hard mismatches
            hard_mismatch_indices_by_para[i] = set(range(len(ep_text.split())))
            continue

        # Both exist — align word by word
        if ep_text is not None and vp_text is not None:
            editor_words = ep_text.split()
            vlm_words = vp_text.split()

            align_discrepancies, para_hard_indices, para_soft_indices = (
                _align_paragraph_words(
                    editor_words, vlm_words, max_lookahead=max_lookahead
                )
            )
            if para_hard_indices:
                hard_mismatch_indices_by_para[i] = para_hard_indices
            if para_soft_indices:
                soft_mismatch_indices_by_para[i] = para_soft_indices

            # Collect missing insertions from discrepancy messages
            para_missing: dict[int, list[str]] = defaultdict(list)
            # Re-run alignment to extract missing_insertions detail
            # (we extract them from the discrepancies directly)
            # We need to re-run to get missing_insertions — refactor to return them
            # Actually _align_paragraph_words doesn't return missing_insertions anymore.
            # Reconstruct them from the discrepancies:
            w_i = 0
            w_j = 0
            ew = editor_words
            vw = vlm_words
            em = len(ew)
            vn = len(vw)
            while w_i < em and w_j < vn:
                if ew[w_i] == vw[w_j]:
                    w_i += 1
                    w_j += 1
                    continue
                ratio = _fuzzy_ratio(ew[w_i], vw[w_j])
                if ratio >= FUZZY_MATCH_THRESHOLD and ew[w_i] != vw[w_j]:
                    w_i += 1
                    w_j += 1
                    continue
                # anchor search
                anchor: tuple[int, int] | None = None
                for total_dist in range(1, 2 * max_lookahead + 1):
                    for di in range(min(total_dist, max_lookahead) + 1):
                        dj = total_dist - di
                        if dj <= max_lookahead and (w_i + di) < em and (w_j + dj) < vn:
                            if (
                                _fuzzy_ratio(ew[w_i + di], vw[w_j + dj])
                                >= FUZZY_MATCH_THRESHOLD
                            ):
                                anchor = (di, dj)
                                break
                    if anchor is not None:
                        break
                if anchor is not None:
                    di, dj = anchor
                    if di == 0 and dj > 0:
                        para_missing[w_i].append(" ".join(vw[w_j : w_j + dj]))
                    w_i += di
                    w_j += dj
                else:
                    w_i += 1
                    w_j += 1
            if w_j < vn:
                para_missing[em].append(" ".join(vw[w_j:]))

            if para_missing:
                missing_words_by_para[i] = dict(para_missing)

            for d in align_discrepancies:
                warnings.append(
                    {
                        "type": d["type"],
                        "para": i,
                        "label": para_label,
                        "msg": d["msg"],
                    }
                )

    return (
        warnings,
        hard_mismatch_indices_by_para,
        soft_mismatch_indices_by_para,
        missing_words_by_para,
        missing_paras_by_idx,
    )


def _format_paragraph_words_html(
    words: list[str],
    hard_flagged: set[int],
    soft_flagged: set[int],
    para_missing: dict[int, list[str]] | None = None,
) -> str:
    """Format a paragraph's words as HTML with hard/soft mismatch highlights.

    Args:
        words: Tokenized words in the paragraph.
        hard_flagged: Word indices with hard mismatches (rendered red bold).
        soft_flagged: Word indices with soft/fuzzy mismatches (rendered orange bold).
        para_missing: Optional mapping of word index to missing VLM segments to
            render inline with ``[MISSING: ...]`` tags.

    Returns:
        Space-joined HTML string for the paragraph.
    """
    formatted_words: list[str] = []

    for w_idx, w in enumerate(words):
        if para_missing and w_idx in para_missing:
            for missing_seg in para_missing[w_idx]:
                escaped_seg = html.escape(missing_seg)
                formatted_words.append(
                    f"<span style='color: red; font-weight: bold; "
                    f"text-decoration: underline;'>[MISSING: {escaped_seg}]</span>"
                )

        escaped_word = html.escape(w)
        # Hard mismatch takes priority over soft
        if w_idx in hard_flagged:
            formatted_words.append(
                f"<span style='color: red; font-weight: bold;'>{escaped_word}</span>"
            )
        elif w_idx in soft_flagged:
            formatted_words.append(
                f"<span style='color: orange; font-weight: bold;'>{escaped_word}</span>"
            )
        else:
            formatted_words.append(escaped_word)

    if para_missing and len(words) in para_missing:
        for missing_seg in para_missing[len(words)]:
            escaped_seg = html.escape(missing_seg)
            formatted_words.append(
                f"<span style='color: red; font-weight: bold; "
                f"text-decoration: underline;'>[MISSING: {escaped_seg}]</span>"
            )

    return " ".join(formatted_words)


def _render_highlighted_text(
    editor_text: str,
    hard_mismatch_indices_by_para: dict[int, set[int]],
    soft_mismatch_indices_by_para: dict[int, set[int]] | None = None,
    missing_words_by_para: dict[int, dict[int, list[str]]] | None = None,
    missing_paras_by_idx: dict[int, str] | None = None,
) -> str:
    """Build an HTML preview with warning words highlighted by severity.

    Hard mismatches are shown in red bold; soft (fuzzy) mismatches in orange
    bold; missing insertions are shown inline with a red underlined [MISSING:]
    tag; missing paragraphs are shown as full-width red blocks.

    A word can only be in one category — hard takes priority over soft.

    Args:
        editor_text: Current editor text with blank-line paragraph separators.
        hard_mismatch_indices_by_para: Maps para index → set of hard-mismatch word indices.
        soft_mismatch_indices_by_para: Maps para index → set of soft-mismatch word indices.
        missing_words_by_para: Maps para index → {word_idx: [missing_vlm_strings]}.
        missing_paras_by_idx: Maps para index → VLM paragraph text for orphan VLM paras.

    Returns:
        HTML string suitable for st.markdown(..., unsafe_allow_html=True).
    """
    if soft_mismatch_indices_by_para is None:
        soft_mismatch_indices_by_para = {}
    if missing_words_by_para is None:
        missing_words_by_para = {}
    if missing_paras_by_idx is None:
        missing_paras_by_idx = {}

    editor_paras = _split_paragraphs(editor_text)
    if not editor_paras and not missing_paras_by_idx:
        return "<p style='color: #888; font-style: italic; margin: 0;'>No text to preview.</p>"

    html_paragraphs: list[str] = []
    total_slots = max(
        len(editor_paras),
        (max(missing_paras_by_idx.keys()) + 1) if missing_paras_by_idx else 0,
    )

    for i in range(total_slots):
        # Render missing (orphan VLM) paragraph block
        if i in missing_paras_by_idx:
            escaped_vp = html.escape(missing_paras_by_idx[i])
            html_paragraphs.append(
                f"<p style='margin: 0 0 10px 0; line-height: 1.5; color: red; "
                f"font-weight: bold; background: #fff0f0; padding: 4px 8px; "
                f"border-left: 3px solid red;'>[MISSING PARAGRAPH: {escaped_vp}]</p>"
            )

        if i < len(editor_paras):
            para = editor_paras[i]
            para_html = _format_paragraph_words_html(
                para.split(),
                hard_mismatch_indices_by_para.get(i, set()),
                soft_mismatch_indices_by_para.get(i, set()),
                para_missing=missing_words_by_para.get(i, {}),
            )
            html_paragraphs.append(
                f"<p style='margin: 0 0 10px 0; line-height: 1.5;'>{para_html}</p>"
            )

    return f"<div style='font-size: 1.15rem; line-height: 1.6;'>{''.join(html_paragraphs)}</div>"


def _render_editable_highlighted_html(
    editor_text: str,
    hard_mismatch_indices_by_para: dict[int, set[int]],
    soft_mismatch_indices_by_para: dict[int, set[int]] | None = None,
) -> str:
    """Build inner HTML for the editable Final text box without missing tags.

    Paragraphs are joined by ``\\n\\n`` so a ``white-space: pre-wrap``
    ``contenteditable`` container renders blank-line paragraph breaks and
    round-trips cleanly back to plain text.
    """
    if soft_mismatch_indices_by_para is None:
        soft_mismatch_indices_by_para = {}

    editor_paras = _split_paragraphs(editor_text)
    if not editor_paras:
        return ""

    formatted_paras = [
        _format_paragraph_words_html(
            para.split(),
            hard_mismatch_indices_by_para.get(i, set()),
            soft_mismatch_indices_by_para.get(i, set()),
            para_missing=None,
        )
        for i, para in enumerate(editor_paras)
    ]
    return "\n\n".join(formatted_paras)


_EDITABLE_HTML_BOX_CSS = """
.ocr-editable-box {
    box-sizing: border-box;
    width: 100%;
    height: 550px;
    overflow-y: auto;
    padding: 0.75rem;
    border: 1px solid var(--st-border-color, rgba(49, 51, 63, 0.2));
    border-radius: var(--st-base-radius, 0.5rem);
    background-color: var(--st-secondary-background-color, #f0f2f6);
    color: var(--st-text-color, inherit);
    font-family: var(--st-font, inherit);
    font-size: 1.15rem;
    line-height: 1.6;
    white-space: pre-wrap;
    word-wrap: break-word;
    outline: none;
}
.ocr-editable-box:focus {
    border-color: var(--st-primary-color, #ff4b4b);
    box-shadow: 0 0 0 1px var(--st-primary-color, #ff4b4b);
}
.ocr-editable-box:empty::before {
    content: attr(data-placeholder);
    color: #888;
    font-style: italic;
    pointer-events: none;
}
"""

_EDITABLE_HTML_BOX_JS = """
export default function(component) {
    const { data, parentElement, setTriggerValue } = component;

    let editor = parentElement.querySelector('.ocr-editable-box');
    if (!editor) {
        editor = document.createElement('div');
        editor.className = 'ocr-editable-box';
        editor.contentEditable = 'true';
        editor.spellcheck = false;
        editor.setAttribute('role', 'textbox');
        editor.setAttribute('aria-multiline', 'true');
        editor.setAttribute('data-placeholder', 'Enter or edit final text...');
        parentElement.appendChild(editor);

        const extractPlainText = (root) => {
            const walk = (node) => {
                if (node.nodeType === Node.TEXT_NODE) {
                    return node.nodeValue || '';
                }
                if (node.nodeType !== Node.ELEMENT_NODE) {
                    return '';
                }
                const tag = node.tagName;
                if (tag === 'BR') {
                    return '\\n';
                }
                let out = '';
                for (const child of node.childNodes) {
                    out += walk(child);
                }
                if ((tag === 'DIV' || tag === 'P') && node.previousSibling) {
                    if (!out.startsWith('\\n')) {
                        out = '\\n' + out;
                    }
                }
                return out;
            };
            let text = walk(root).replace(/\\r\\n?/g, '\\n');
            if (text === '\\n' && root.querySelectorAll('*').length <= 1) {
                return '';
            }
            return text;
        };

        const insertPlainText = (text) => {
            if (document.queryCommandSupported && document.queryCommandSupported('insertText')) {
                if (document.execCommand('insertText', false, text)) {
                    return;
                }
            }
            const sel = window.getSelection();
            if (!sel || !sel.rangeCount) return;
            const range = sel.getRangeAt(0);
            range.deleteContents();
            const textNode = document.createTextNode(text);
            range.insertNode(textNode);
            range.setStartAfter(textNode);
            range.collapse(true);
            sel.removeAllRanges();
            sel.addRange(range);
        };

        const commit = () => {
            const current = extractPlainText(editor);
            const last = editor.dataset.lastCommitted ?? '';
            if (current !== last) {
                editor.dataset.lastCommitted = current;
                setTriggerValue('text', current);
            }
        };

        editor.addEventListener('blur', commit);

        editor.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                commit();
                return;
            }
            if (e.key === 'Enter' && !e.ctrlKey && !e.metaKey && !e.altKey) {
                e.preventDefault();
                insertPlainText('\\n');
            }
        });

        editor.addEventListener('paste', (e) => {
            e.preventDefault();
            const text = (e.clipboardData || window.clipboardData)
                .getData('text/plain')
                .replace(/\\r\\n?/g, '\\n');
            insertPlainText(text);
        });
    }

    const incomingPlain = (data && data.plain) ? data.plain : '';
    const incomingHtml = (data && data.html) ? data.html : '';
    const isFocused = (parentElement.activeElement === editor) || (document.activeElement === editor);

    if (!isFocused || editor.dataset.lastCommitted !== incomingPlain) {
        if (!isFocused) {
            editor.innerHTML = incomingHtml;
            editor.dataset.lastCommitted = incomingPlain;
        }
    }
}
"""

_EDITABLE_HTML_BOX = (
    st.components.v2.component(
        "ocr_editable_html_box",
        css=_EDITABLE_HTML_BOX_CSS,
        js=_EDITABLE_HTML_BOX_JS,
    )
    if hasattr(st, "components") and hasattr(st.components, "v2")
    else None
)


# ---------------------------------------------------------------------------
# Staging file helpers
# ---------------------------------------------------------------------------


def _staging_path(output_path: str) -> Path:
    """Return the staging JSONL path derived from the final output path.

    Args:
        output_path: Path to the final OCRDocument JSON file.

    Returns:
        Path to the staging .jsonl file in the same directory.
    """
    p = Path(output_path)
    return p.parent / (p.stem + "_staging.jsonl")


def _sync_staging_file(confirmed_pages: list[dict | None], output_path: str) -> None:
    """Rewrite staging file with all confirmed pages in JSON Lines format.

    Args:
        confirmed_pages: List of page dicts (None for unconfirmed slots).
        output_path: Path to the final OCRDocument JSON (determines staging path).
    """
    staging = _staging_path(output_path)
    staging.parent.mkdir(parents=True, exist_ok=True)
    with staging.open("w", encoding="utf-8") as f:
        for page_dict in confirmed_pages:
            if page_dict is not None:
                f.write(json.dumps(page_dict, ensure_ascii=False) + "\n")


def _load_staging(output_path: str) -> list[OCRPage]:
    """Load confirmed pages from the staging JSONL file or existing output JSON.

    Args:
        output_path: Path to the final OCRDocument JSON (determines staging path).

    Returns:
        List of OCRPage objects deserialized from the staging file or existing JSON.
    """
    staging = _staging_path(output_path)
    pages: list[OCRPage] = []
    if staging.exists():
        for line in staging.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    pages.append(OCRPage.model_validate_json(line))
                except Exception:
                    pass
        if pages:
            return pages

    # Fallback to existing output JSON if staging doesn't exist
    out_p = Path(output_path)
    if out_p.exists():
        try:
            content = out_p.read_text(encoding="utf-8").strip()
            if content:
                loaded = json.loads(content)
                if isinstance(loaded, list):
                    for item in loaded:
                        try:
                            pages.append(OCRPage.model_validate(item))
                        except Exception:
                            pass
        except Exception:
            pass
    return pages


# ---------------------------------------------------------------------------
# Document saving helpers
# ---------------------------------------------------------------------------


def _save_or_update_ocr_page_in_json(
    output_path: str,
    page_obj: OCRPage,
    overwrite_existing: bool = False,
) -> tuple[bool, str]:
    """Append or overwrite a single OCRPage in the output JSON file.

    If the file does not exist, it is created with [page_obj].
    If the file exists:
      - If a page with matching pdf_page_number exists and overwrite_existing is False,
        returns (False, "already_exists").
      - If matching page exists and overwrite_existing is True, it overwrites that entry
        and returns (True, "overwritten").
      - If no matching page exists, it appends page_obj, sorts pages by pdf_page_number,
        and returns (True, "appended").

    Args:
        output_path: Path to the target JSON file.
        page_obj: OCRPage to insert or update.
        overwrite_existing: Whether to allow overwriting if pdf_page_number exists.

    Returns:
        (success, action_or_reason)
    """
    out_p = Path(output_path)
    existing_pages: list[dict] = []
    if out_p.exists():
        try:
            content = out_p.read_text(encoding="utf-8").strip()
            if content:
                loaded = json.loads(content)
                if isinstance(loaded, list):
                    existing_pages = loaded
        except Exception:
            existing_pages = []

    # Check if page already exists
    match_idx = -1
    for idx, p in enumerate(existing_pages):
        if isinstance(p, dict) and p.get("pdf_page_number") == page_obj.pdf_page_number:
            match_idx = idx
            break

    if match_idx != -1:
        if not overwrite_existing:
            return False, "already_exists"
        existing_pages[match_idx] = page_obj.model_dump()
        action = "overwritten"
    else:
        existing_pages.append(page_obj.model_dump())
        action = "appended"

    # Sort pages by pdf_page_number
    existing_pages.sort(key=lambda p: p.get("pdf_page_number", 0))

    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(
        json.dumps(existing_pages, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return True, action


def _get_image_dimensions(
    sample_image_path: Path | str | None = None,
    images: list[np.ndarray] | None = None,
) -> tuple[int | None, int | None]:
    """Retrieve image dimensions (height, width) from a path or image array."""
    if images and len(images) > 0 and hasattr(images[0], "shape"):
        h, w = images[0].shape[:2]
        return int(h), int(w)
    if sample_image_path:
        p = Path(sample_image_path)
        if p.exists():
            try:
                with Image.open(p) as img:
                    w, h = img.size
                    return int(h), int(w)
            except Exception:
                pass
    return None, None


def _load_image_from_disk(image_path: Path | str) -> np.ndarray:
    """Load an image file from disk and convert BGR to RGB."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _perform_save_document(
    output_path: str,
    source_name: str,
    language: str,
    sample_image_path: Path | str | None = None,
    images: list[np.ndarray] | None = None,
) -> tuple[int, Path]:
    """Write confirmed pages to an intermediate OCRPage list JSON file.

    The file contains a JSON array of :class:`OCRPage` model dicts, which can
    be reloaded with ``[OCRPage(**p) for p in json.loads(text)]``.
    The staging file is also synced after writing.

    Args:
        output_path: Destination path for the intermediate pages JSON.
        source_name: Source PDF filename (unused; kept for API compatibility).
        language: Document language code (unused; kept for API compatibility).
        sample_image_path: Unused; kept for API compatibility.
        images: Unused; kept for API compatibility.

    Returns:
        A 2-tuple of (n_saved_pages, out_path).
    """
    confirmed_pages_raw = st.session_state.get("ocr_confirmed_pages", [])
    confirmed_page_objs = [OCRPage(**p) for p in confirmed_pages_raw if p is not None]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pages_data = [p.model_dump() for p in confirmed_page_objs]
    out_path.write_text(
        json.dumps(pages_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Sync staging file
    _sync_staging_file(confirmed_pages_raw, output_path)

    return len(confirmed_page_objs), out_path


def _resolve_export_image_path(
    page: OCRPage,
    document_name: str,
    language: str,
    image_paths: list[Path] | None = None,
    export_dir: str | Path | None = None,
) -> str:
    """Resolve the relative ``<language>/<filename>`` image path for PPOCR export.

    Prefers matching the original on-disk image filename (from *image_paths* or
    the document/export directory) by ``page.pdf_page_number`` so the exported
    path matches the original image name rather than the offset-adjusted
    ``document_page_number``. Falls back to :func:`make_image_path` with
    ``page.pdf_page_number``.
    """
    candidates = list(image_paths) if image_paths else []
    if not candidates:
        search_dirs: list[Path] = []
        if document_name:
            search_dirs.append(Path(document_name) / language)
        if export_dir:
            search_dirs.append(Path(export_dir))
        for d in search_dirs:
            if d.is_dir():
                candidates = sorted(d.glob(f"{document_name}.{page.pdf_page_number:03d}*"))
                if candidates:
                    break

    for p in candidates:
        parts = p.stem.split(".")
        if len(parts) >= 2:
            try:
                p_num = int(parts[-1].split("_")[0])
                if p_num == page.pdf_page_number:
                    return f"{language}/{p.name}"
            except ValueError:
                continue

    return make_image_path(document_name, language, page.pdf_page_number)


def _perform_export_ppocr(
    export_dir: str,
    document_name: str,
    language: str,
    overwrite: bool = False,
    image_paths: list[Path] | None = None,
) -> tuple[int, Path, Path]:
    """Export confirmed pages as PPOCRLabel Label.txt and fileState.txt.

    Image paths are resolved from the original on-disk image filenames (or
    :func:`make_image_path` using ``pdf_page_number``) rather than the
    offset-adjusted ``document_page_number``. The header paragraph (first
    paragraph) is parsed for ``document_page_number`` and ``align_to``, and is
    omitted from the exported Label.txt annotations along with its bounding box.

    Args:
        export_dir: Directory in which to write ``Label.txt`` and
            ``fileState.txt``.
        document_name: Document stem (from ``st.session_state.document_name``).
        language: Language folder (from ``st.session_state.document_language``).
        overwrite: When ``True`` existing output files are overwritten without
            prompting.
        image_paths: Optional list of original on-disk image paths.

    Returns:
        A 3-tuple of ``(n_pages, label_path, state_path)``.

    Raises:
        FileExistsError: If either output file exists and *overwrite* is
            ``False``.
    """
    confirmed_pages_raw = st.session_state.get("ocr_confirmed_pages", [])
    confirmed_page_objs = [OCRPage(**p) for p in confirmed_pages_raw if p is not None]

    ppocr_pages: list[PPOCRLabelPage] = []
    for page in sorted(confirmed_page_objs, key=lambda p: p.pdf_page_number):
        # Extract header info if present in first paragraph
        parsed_doc_page, parsed_align_to = None, None
        has_header = False
        if page.paragraphs:
            p_doc, p_align = parse_header_paragraph(
                page.paragraphs[0].text, language=language
            )
            if p_doc is not None:
                has_header = True
                parsed_doc_page = p_doc
                parsed_align_to = p_align

        doc_page_to_use = (
            parsed_doc_page
            if parsed_doc_page is not None
            else page.document_page_number
        )
        img_path = _resolve_export_image_path(
            page,
            document_name,
            language,
            image_paths=image_paths,
            export_dir=export_dir,
        )
        ppocr_page = ocr_page_to_ppocr(
            page,
            img_path,
            document_page_number=doc_page_to_use,
            align_to=parsed_align_to,
            skip_first_paragraph=has_header,
        )
        ppocr_pages.append(ppocr_page)

    out_dir = Path(export_dir)
    label_path = out_dir / "Label.txt"
    state_path = out_dir / "fileState.txt"

    write_label_txt(ppocr_pages, label_path, overwrite=overwrite)
    write_file_state_txt(ppocr_pages, state_path, overwrite=overwrite)

    return len(ppocr_pages), label_path, state_path


def _execute_page_save(
    page_data: dict,
    output_path: str,
    index: int,
    total: int,
) -> None:
    """Execute the persistence of a confirmed OCR page object."""
    page_obj = OCRPage(**page_data)
    confirmed_flags = st.session_state.ocr_confirmed_flags
    confirmed_flags[index] = True
    st.session_state.ocr_confirmed_flags = confirmed_flags
    st.session_state.ocr_confirmed_pages[index] = page_obj.model_dump()

    # Rewrite staging file with all currently confirmed pages
    try:
        _sync_staging_file(st.session_state.ocr_confirmed_pages, output_path)
    except Exception as e:
        st.error(f"Could not write staging file: {e}")

    # Save or update in target output_path (ocr_result.json)
    saved_ok, action_or_reason = _save_or_update_ocr_page_in_json(
        output_path=output_path,
        page_obj=page_obj,
        overwrite_existing=False,
    )

    if not saved_ok and action_or_reason == "already_exists":
        st.session_state["pending_page_overwrite"] = page_obj.model_dump()
        st.session_state["show_page_overwrite_dialog"] = True
        st.rerun()
    else:
        st.session_state["save_alert"] = (
            "success",
            f"Page {page_obj.pdf_page_number} saved ({action_or_reason}) to `{output_path}`.",
        )
        if index < total - 1:
            st.session_state.ocr_current_index = index + 1
        st.rerun()


# Overwrite confirmation dialog (requires Streamlit >= 1.29 for st.dialog)
if hasattr(st, "dialog"):

    @st.dialog("⚠️ Page Number Mismatch")
    def _confirm_page_mismatch_dialog(
        calculated_page: int,
        parsed_page: int,
        pending_data: dict,
        output_path: str,
        index: int,
        total: int,
    ) -> None:
        """Modal dialog to warn and resolve page number mismatch."""
        st.warning(
            f"Page mismatch detected! Calculated document page is **{calculated_page}**, "
            f"but header in text parsed as **{parsed_page}**."
        )
        st.write("Which page number should be used for this page?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                f"Use Parsed ({parsed_page})", type="primary", use_container_width=True
            ):
                st.session_state.pop("show_page_mismatch_dialog", None)
                st.session_state.pop("pending_page_mismatch", None)
                pending_data["document_page_number"] = parsed_page
                _execute_page_save(pending_data, output_path, index, total)
        with col2:
            if st.button(
                f"Use Calculated ({calculated_page})", use_container_width=True
            ):
                st.session_state.pop("show_page_mismatch_dialog", None)
                st.session_state.pop("pending_page_mismatch", None)
                pending_data["document_page_number"] = calculated_page
                _execute_page_save(pending_data, output_path, index, total)

    @st.dialog("⚠️ Overwrite Page in OCR Result")
    def _confirm_overwrite_page_dialog(
        output_path: str,
        page_dict: dict,
    ) -> None:
        """Modal dialog to confirm overwriting an existing page in ocr_result.json."""
        p_num = page_dict.get("pdf_page_number")
        st.warning(f"Page {p_num} already exists in `{output_path}`.")
        st.write("Do you want to overwrite this page's entry in the OCR result file?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "Yes, Overwrite Page", type="primary", use_container_width=True
            ):
                page_obj = OCRPage(**page_dict)
                ok, action = _save_or_update_ocr_page_in_json(
                    output_path, page_obj, overwrite_existing=True
                )
                msg = f"Page {p_num} overwritten in `{output_path}`."
                st.session_state["save_alert"] = ("success", msg)
                st.session_state.pop("show_page_overwrite_dialog", None)
                st.session_state.pop("pending_page_overwrite", None)
                st.rerun()
        with col2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.pop("show_page_overwrite_dialog", None)
                st.session_state.pop("pending_page_overwrite", None)
                st.rerun()

    @st.dialog("⚠️ File Already Exists")
    def _confirm_overwrite_dialog(
        output_path: str,
        source_name: str,
        language: str,
        sample_image_path: Path | str | None = None,
        images: list[np.ndarray] | None = None,
    ) -> None:
        """Modal dialog to confirm overwriting an existing pages JSON.

        Args:
            output_path: Path to the existing file.
            source_name: Source PDF filename.
            language: Document language code.
            sample_image_path: Path to an extracted image.
            images: Original extracted images.
        """
        st.warning(f"File `{output_path}` already exists.")
        st.write("Do you want to overwrite the existing file?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, Overwrite", type="primary", use_container_width=True):
                n_pages, p = _perform_save_document(
                    output_path,
                    source_name,
                    language,
                    sample_image_path=sample_image_path,
                    images=images,
                )
                msg = f"Saved {n_pages} page(s) to `{p}`."
                st.session_state["save_alert"] = ("success", msg)
                st.rerun()
        with col2:
            if st.button("Cancel", use_container_width=True):
                st.rerun()

    @st.dialog("⚠️ Export Files Already Exist")
    def _confirm_overwrite_export_dialog(
        export_dir: str,
        document_name: str,
        language: str,
        image_paths: list[Path] | None = None,
    ) -> None:
        """Modal dialog to confirm overwriting existing PPOCRLabel export files.

        Args:
            export_dir: Directory containing existing Label.txt / fileState.txt.
            document_name: Document stem.
            language: Language folder.
            image_paths: Optional list of original on-disk image paths.
        """
        st.warning(f"`Label.txt` or `fileState.txt` already exists in `{export_dir}`.")
        st.write("Do you want to overwrite the existing export?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, Overwrite", type="primary", use_container_width=True):
                try:
                    n_pages, label_p, state_p = _perform_export_ppocr(
                        export_dir,
                        document_name,
                        language,
                        overwrite=True,
                        image_paths=image_paths,
                    )
                    msg = f"Exported {n_pages} page(s) to `{label_p}` and `{state_p}`."
                    st.session_state["export_alert"] = ("success", msg)
                except Exception as exc:
                    st.session_state["export_alert"] = ("error", str(exc))
                st.rerun()
        with col2:
            if st.button("Cancel", use_container_width=True):
                st.rerun()


def _handle_export_request(
    export_dir: str,
    document_name: str,
    language: str,
    image_paths: list[Path] | None = None,
) -> None:
    """Handle a PPOCRLabel export request with overwrite protection.

    Checks whether ``Label.txt`` or ``fileState.txt`` already exist in
    *export_dir*.  If either does and overwrite has not been approved, shows the
    confirmation dialog.  Otherwise proceeds with the export.

    Args:
        export_dir: Directory in which to write the export files.
        document_name: Document stem.
        language: Language folder.
        image_paths: Optional list of original on-disk image paths.
    """
    label_p = Path(export_dir) / "Label.txt"
    state_p = Path(export_dir) / "fileState.txt"
    files_exist = label_p.exists() or state_p.exists()

    if files_exist:
        _confirm_overwrite_export_dialog(
            export_dir, document_name, language, image_paths=image_paths
        )
    else:
        try:
            n_pages, label_out, state_out = _perform_export_ppocr(
                export_dir,
                document_name,
                language,
                overwrite=True,
                image_paths=image_paths,
            )
            msg = f"Exported {n_pages} page(s) → `{label_out}` and `{state_out}`."
            st.session_state["export_alert"] = ("success", msg)
        except Exception as exc:
            st.session_state["export_alert"] = ("error", str(exc))
        st.rerun()


def _handle_save_request(
    output_path: str,
    source_name: str,
    language: str,
    sample_image_path: Path | str | None = None,
    images: list[np.ndarray] | None = None,
) -> None:
    """Handle a save request, checking for existing files and prompting overwrite.

    Args:
        output_path: Destination path for the OCRDocument JSON.
        source_name: Source PDF filename.
        language: Document language code.
        sample_image_path: Path to an extracted image.
        images: Original extracted images.
    """
    out_p = Path(output_path)
    if out_p.exists() and not st.session_state.get("overwrite_approved", False):
        if hasattr(st, "dialog"):
            _confirm_overwrite_dialog(
                output_path,
                source_name,
                language,
                sample_image_path=sample_image_path,
                images=images,
            )
        else:
            # Fallback for environments where st.dialog is unavailable
            st.session_state["show_inline_overwrite_warning"] = True
            st.rerun()
    else:
        st.session_state["overwrite_approved"] = False
        n_pages, p = _perform_save_document(
            output_path,
            source_name,
            language,
            sample_image_path=sample_image_path,
            images=images,
        )
        msg = f"Saved {n_pages} page(s) to `{p}`."
        st.session_state["save_alert"] = ("success", msg)
        st.rerun()


# ---------------------------------------------------------------------------
# OCR execution
# ---------------------------------------------------------------------------


def _process_and_ocr_for_index(
    index: int,
    image_input: np.ndarray | Path | str | list[np.ndarray],
    pages: list[int],
    pipeline_steps: list[dict],
    page_adjustment: int,
    psm_config: str,
    min_width: int,
    min_height: int,
) -> dict:
    """Apply preprocessing and run both OCR engines on a single image.

    The preprocessing pipeline is applied to produce ``preprocessed_image``;
    Tesseract runs on the preprocessed image while VLM runs on the original
    raw image. If the pipeline is empty the raw image is used for Tesseract.

    Args:
        index: Index into pages list.
        image_input: Image array, image Path/str on disk, or list of images.
        pages: List of PDF page numbers corresponding to index.
        pipeline_steps: Serialized PreprocessPipeline steps list.
        page_adjustment: Offset added to PDF page number to get document page.
        psm_config: Tesseract PSM configuration string (e.g. "--psm 3").
        min_width: Minimum bounding-box width for Tesseract filtering.
        min_height: Minimum bounding-box height for Tesseract filtering.

    Returns:
        Dict with keys:
          - preprocessed_image: np.ndarray after pipeline
          - tess_result: OCRPage or None
          - tess_annotated: PIL Image or None
          - tess_error: str or None
          - vlm_result: OCRPage or None
          - vlm_error: str or None
          - baseline: str seed for editor
    """
    pdf_page = pages[index]
    doc_page = pdf_page + page_adjustment
    page_name = f"{pdf_page:03d}"

    if isinstance(image_input, (str, Path)):
        raw_image = _load_image_from_disk(image_input)
    elif isinstance(image_input, list):
        raw_image = image_input[index]
    else:
        raw_image = image_input

    # Parallel execution: run (Preprocessing + Tesseract) and (VLM OCR) concurrently
    def _prep_and_tess() -> tuple[
        np.ndarray,
        OCRPage | None,
        Image.Image | None,
        Image.Image | None,
        dict | None,
        str | None,
    ]:
        try:
            prep = PreprocessPipeline(pipeline_steps).apply(raw_image)
        except Exception:
            prep = raw_image

        try:
            tess = TesseractOCR(
                lang="vie", confidence_threshold=LOW_CONFIDENCE_THRESHOLD
            )
            res = tess.run(
                prep,
                pdf_page_number=pdf_page,
                document_page_number=doc_page,
                page_name=page_name,
                config=psm_config,
                min_width=min_width,
                min_height=min_height,
            )
            annotated_words = tess.annotate_image()
            annotated_paras = tess.annotate_paragraphs()
            filtered_raw = tess.filtered_raw
            return prep, res, annotated_words, annotated_paras, filtered_raw, None
        except Exception as e:
            return prep, None, None, None, None, str(e)

    def _run_vlm() -> tuple[OCRPage | None, str | None]:
        try:
            vlm = VLMOCR(model=VLM_MODEL)
            res = vlm.run(
                raw_image,
                pdf_page_number=pdf_page,
                document_page_number=doc_page,
                page_name=page_name,
            )
            return res, None
        except Exception as e:
            return None, str(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        tess_future = executor.submit(_prep_and_tess)
        vlm_future = executor.submit(_run_vlm)

        (
            preprocessed_image,
            tess_result,
            tess_annotated,
            tess_annotated_paras,
            tess_filtered_raw,
            tess_error,
        ) = tess_future.result()
        vlm_result, vlm_error = vlm_future.result()

    # Seed editor with Tesseract baseline (or VLM fallback, then empty)
    if tess_result is not None and tess_result.paragraphs:
        baseline = "\n\n".join(p.text for p in tess_result.paragraphs)
    elif vlm_result is not None and vlm_result.paragraphs:
        baseline = "\n\n".join(p.text for p in vlm_result.paragraphs)
    else:
        baseline = ""

    return {
        "preprocessed_image": preprocessed_image,
        "tess_result": tess_result,
        "tess_annotated": tess_annotated,
        "tess_annotated_paras": tess_annotated_paras,
        "tess_filtered_raw": tess_filtered_raw,
        "tess_error": tess_error,
        "vlm_result": vlm_result,
        "vlm_error": vlm_error,
        "baseline": baseline,
    }


# ---------------------------------------------------------------------------
# Prefetch live status display
# ---------------------------------------------------------------------------


if hasattr(st, "fragment"):

    @st.fragment(run_every="1s")
    def _render_prefetch_status(next_idx: int, total_pages: int) -> None:
        """Render live auto-updating prefetch indicator."""
        with _PREFETCH_LOCK:
            is_running = next_idx in _PREFETCH_IN_PROGRESS
            is_ready = next_idx in _PREFETCH_CACHE or next_idx in st.session_state.get(
                "ocr_page_cache", {}
            )
        if is_running:
            st.caption(
                f"⚡ Background OCR running for next image (Image {next_idx + 1} of {total_pages})..."
            )
        elif is_ready:
            st.caption(
                f"✅ Next image (Image {next_idx + 1}) OCR pre-computed and ready."
            )

else:

    def _render_prefetch_status(next_idx: int, total_pages: int) -> None:
        """Render static prefetch indicator (fallback for older Streamlit)."""
        with _PREFETCH_LOCK:
            is_running = next_idx in _PREFETCH_IN_PROGRESS
            is_ready = next_idx in _PREFETCH_CACHE or next_idx in st.session_state.get(
                "ocr_page_cache", {}
            )
        if is_running:
            st.caption(
                f"⚡ Background OCR running for next image (Image {next_idx + 1} of {total_pages})..."
            )
        elif is_ready:
            st.caption(
                f"✅ Next image (Image {next_idx + 1}) OCR pre-computed and ready."
            )


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for the OCR Review page (page 3).

    Renders the full OCR review workflow: config settings, per-image OCR with
    background prefetch, side-by-side review with mismatch highlighting, and
    finalize/save controls.
    """
    st.header("3. OCR Review")

    # -----------------------------------------------------------------------
    # OCR Settings expander (top of page; replaces sidebar controls)
    # -----------------------------------------------------------------------
    preprocess_steps: list[dict] = st.session_state.get("preprocess_steps", [])

    # Pipeline change detection: clear cache if pipeline changed
    sig = hashlib.md5(json.dumps(preprocess_steps, sort_keys=True).encode()).hexdigest()
    if st.session_state.get("ocr_preprocess_signature") != sig:
        st.session_state.ocr_preprocess_signature = sig
        st.session_state.pop("ocr_page_cache", None)
        st.session_state.pop("ocr_images_signature", None)
        with _PREFETCH_LOCK:
            _PREFETCH_CACHE.clear()
            _PREFETCH_IN_PROGRESS.clear()
        # Do NOT clear confirmed pages

    with st.expander(
        "⚙️ Settings",
        expanded=not bool(st.session_state.get("ocr_page_cache")),
    ):
        cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
        with cfg_col1:
            psm_config = st.selectbox(
                "Page segmentation method",
                list(PSM_NOTES),
                format_func=lambda c: f"{c} - {PSM_NOTES[c]}",
                key="psm_config",
            )

            language = st.radio(
                "Language",
                options=["vi", "ch"],
                index=0,
                format_func=lambda x: "Quốc ngữ (vi)" if x == "vi" else "Hán Nôm (ch)",
                horizontal=True,
                key="document_language",
            )

        with cfg_col2:
            min_width = int(
                st.number_input(
                    "Min bbox width (px)",
                    min_value=0,
                    value=1,
                    step=1,
                    key="bbox_min_width",
                )
            )
            min_height = int(
                st.number_input(
                    "Min bbox height (px)",
                    min_value=0,
                    value=5,
                    step=1,
                    key="bbox_min_height",
                )
            )
        with cfg_col3:
            output_filename = st.text_input(
                "Output JSON filename",
                value=st.session_state.get("ocr_output_filename", "ocr_result.json"),
                key="ocr_output_filename",
                help="Name of the JSON file to save inside the document directory (document_name/language/).",
            )
            if "document_name" not in st.session_state:
                st.session_state.document_name = ""
            st.text_input(
                "Source (PDF filename)",
                key="document_name",
                help="Base folder and document name for extracted images",
            )

        is_modified = st.session_state.get(
            "preprocess_modified", bool(preprocess_steps)
        )
        if is_modified:
            st.caption(
                f"ℹ️ Preprocessing pipeline: **Active ({len(preprocess_steps)} step{'s' if len(preprocess_steps) != 1 else ''})**"
            )
        else:
            st.caption("ℹ️ Preprocessing pipeline: **None (using raw images)**")

    # -----------------------------------------------------------------------
    # Document and image discovery from disk
    # -----------------------------------------------------------------------
    document_name = st.session_state.get("document_name", "").strip()
    language = st.session_state.get("document_language", "vi")

    destination_dir = Path(document_name) / language if document_name else None
    default_export_dir = str(destination_dir) if destination_dir else "output"
    output_filename_clean = (
        output_filename.strip() if output_filename.strip() else "ocr_result.json"
    )
    output_path = str(Path(default_export_dir) / output_filename_clean)
    image_paths: list[Path] = []
    pages: list[int] = st.session_state.get("pages", [])

    if destination_dir and destination_dir.is_dir():
        if pages:
            for page_num in pages:
                pattern = f"{document_name}.{page_num:03d}*.jpg"
                matched = sorted(destination_dir.glob(pattern))
                if matched:
                    image_paths.extend(matched)
        if not image_paths:
            all_jpgs = sorted(destination_dir.glob(f"{document_name}.*.jpg"))
            discovered_pages: list[int] = []
            for p in all_jpgs:
                parts = p.stem.split(".")
                if len(parts) >= 2:
                    try:
                        p_num = int(parts[-1].split("_")[0])
                        discovered_pages.append(p_num)
                        image_paths.append(p)
                    except ValueError:
                        pass
            if image_paths and not pages:
                pages = discovered_pages
                st.session_state.pages = pages

    legacy_images: list[np.ndarray] = st.session_state.get("images", [])
    if not image_paths and legacy_images:
        total = len(legacy_images)
        if not pages:
            pages = list(range(1, total + 1))
            st.session_state.pages = pages
        sample_image_path = None
    elif image_paths:
        total = len(image_paths)
        sample_image_path = image_paths[0]
    else:
        if not document_name:
            st.warning(
                "⚠️ No document specified. Extract a document on Page 1 "
                "(1_upload_document) or enter the document name in the OCR Configuration above."
            )
        elif destination_dir and not destination_dir.is_dir():
            st.warning(
                f"⚠️ Directory `{destination_dir}` does not exist. "
                f"Extract pages for '{document_name}' on Page 1 (1_upload_document) "
                f"or check that the language ('{language}') and document name match your extracted files."
            )
        else:
            st.warning(
                f"⚠️ No matching extracted images (`{document_name}.*.jpg`) found in `{destination_dir}`. "
                "Extract PDF pages on Page 1 (1_upload_document) before running OCR."
            )
        return

    images_sig = f"{document_name}_{language}_{total}" if image_paths else str(total)

    # Reset cache if image count or document changed
    if (
        "ocr_images_signature" not in st.session_state
        or st.session_state.ocr_images_signature != images_sig
    ):
        st.session_state.ocr_images_signature = images_sig
        st.session_state.ocr_page_cache = {}
        with _PREFETCH_LOCK:
            _PREFETCH_CACHE.clear()
            _PREFETCH_IN_PROGRESS.clear()

    # Session state init
    if "ocr_page_cache" not in st.session_state:
        st.session_state.ocr_page_cache = {}
    if "ocr_current_index" not in st.session_state:
        st.session_state.ocr_current_index = 0
    if "ocr_confirmed_flags" not in st.session_state:
        st.session_state.ocr_confirmed_flags = [False] * total
    if "ocr_confirmed_pages" not in st.session_state:
        st.session_state.ocr_confirmed_pages = [None] * total
        # Attempt to restore from staging if existing
        staged_pages = _load_staging(output_path)
        for sp in staged_pages:
            if sp.pdf_page_number in pages:
                p_idx = pages.index(sp.pdf_page_number)
                st.session_state.ocr_confirmed_pages[p_idx] = sp.model_dump()
                st.session_state.ocr_confirmed_flags[p_idx] = True
                sp_key = f"ocr_editor_widget_{p_idx}"
                if sp_key not in st.session_state:
                    st.session_state[sp_key] = "\n\n".join(
                        para.text for para in sp.paragraphs
                    )

    if "page_adjustment" not in st.session_state:
        st.session_state.page_adjustment = 0

    # Resize lists if image count changed
    while len(st.session_state.ocr_confirmed_flags) < total:
        st.session_state.ocr_confirmed_flags.append(False)
    while len(st.session_state.ocr_confirmed_pages) < total:
        st.session_state.ocr_confirmed_pages.append(None)

    index: int = int(st.session_state.ocr_current_index)
    index = max(0, min(index, total - 1))

    confirmed_flags: list[bool] = st.session_state.ocr_confirmed_flags
    n_confirmed = sum(confirmed_flags)

    # -----------------------------------------------------------------------
    # Save notification banners
    # -----------------------------------------------------------------------
    if "save_alert" in st.session_state:
        level, msg = st.session_state.pop("save_alert")
        if level == "success":
            st.success(msg)
        elif level == "error":
            st.error(msg)

    if "export_alert" in st.session_state:
        level, msg = st.session_state.pop("export_alert")
        if level == "success":
            st.success(msg)
        elif level == "error":
            st.error(msg)

    # Overwrite confirmation dialog for page in ocr_result.json
    if hasattr(st, "dialog") and st.session_state.get(
        "show_page_overwrite_dialog", False
    ):
        pending_page_data = st.session_state.get("pending_page_overwrite")
        if pending_page_data:
            _confirm_overwrite_page_dialog(
                output_path=output_path,
                page_dict=pending_page_data,
            )

    # Page mismatch confirmation dialog
    if hasattr(st, "dialog") and st.session_state.get(
        "show_page_mismatch_dialog", False
    ):
        mismatch_info = st.session_state.get("pending_page_mismatch")
        if mismatch_info:
            _confirm_page_mismatch_dialog(
                calculated_page=mismatch_info["calculated_page"],
                parsed_page=mismatch_info["parsed_page"],
                pending_data=mismatch_info["pending_data"],
                output_path=output_path,
                index=mismatch_info["index"],
                total=mismatch_info["total"],
            )

    # Fallback inline overwrite warning for older Streamlit
    if not hasattr(st, "dialog") and st.session_state.get(
        "show_inline_overwrite_warning", False
    ):
        st.warning(f"File `{output_path}` already exists.")
        col_ow1, col_ow2 = st.columns(2)
        with col_ow1:
            if st.button("⚠️ Confirm Overwrite File", type="primary"):
                st.session_state["show_inline_overwrite_warning"] = False
                st.session_state["overwrite_approved"] = True
                source_name = st.session_state.get("document_name", "document.pdf")
                language = st.session_state.get("document_language", "vi")
                _handle_save_request(
                    output_path=output_path,
                    source_name=source_name,
                    language=language,
                    sample_image_path=sample_image_path,
                    images=legacy_images if not image_paths else None,
                )
        with col_ow2:
            if st.button("Cancel"):
                st.session_state["show_inline_overwrite_warning"] = False
                st.rerun()

    # Fallback inline overwrite warning for single page in ocr_result.json
    if st.session_state.get("show_inline_page_overwrite_warning", False):
        pending_page_data = st.session_state.get("pending_page_overwrite")
        if pending_page_data:
            p_num = pending_page_data.get("pdf_page_number")
            st.warning(
                f"Page {p_num} already exists in `{output_path}`. Confirm overwriting this page?"
            )
            col_pow1, col_pow2 = st.columns(2)
            with col_pow1:
                if st.button("⚠️ Confirm Overwrite Page", type="primary"):
                    st.session_state["show_inline_page_overwrite_warning"] = False
                    page_obj = OCRPage(**pending_page_data)
                    _save_or_update_ocr_page_in_json(
                        output_path, page_obj, overwrite_existing=True
                    )
                    st.session_state["save_alert"] = (
                        "success",
                        f"Page {p_num} overwritten in `{output_path}`.",
                    )
                    st.session_state.pop("pending_page_overwrite", None)
                    st.rerun()
            with col_pow2:
                if st.button("Cancel Overwrite Page"):
                    st.session_state["show_inline_page_overwrite_warning"] = False
                    st.session_state.pop("pending_page_overwrite", None)
                    st.rerun()

    # Fallback inline page mismatch warning
    if st.session_state.get("show_inline_page_mismatch_warning", False):
        mismatch_info = st.session_state.get("pending_page_mismatch")
        if mismatch_info:
            calc_p = mismatch_info["calculated_page"]
            pars_p = mismatch_info["parsed_page"]
            pend_d = mismatch_info["pending_data"]
            idx_p = mismatch_info["index"]
            tot_p = mismatch_info["total"]
            st.warning(
                f"Page mismatch detected! Calculated document page is **{calc_p}**, "
                f"but header in text parsed as **{pars_p}**."
            )
            col_mm1, col_mm2 = st.columns(2)
            with col_mm1:
                if st.button(f"Use Parsed ({pars_p})", type="primary"):
                    st.session_state["show_inline_page_mismatch_warning"] = False
                    st.session_state.pop("pending_page_mismatch", None)
                    pend_d["document_page_number"] = pars_p
                    _execute_page_save(pend_d, output_path, idx_p, tot_p)
            with col_mm2:
                if st.button(f"Use Calculated ({calc_p})"):
                    st.session_state["show_inline_page_mismatch_warning"] = False
                    st.session_state.pop("pending_page_mismatch", None)
                    pend_d["document_page_number"] = calc_p
                    _execute_page_save(pend_d, output_path, idx_p, tot_p)

    # -----------------------------------------------------------------------
    # Header: page info & navigation controls
    # -----------------------------------------------------------------------
    pdf_page = pages[index]
    editor_key = f"ocr_editor_widget_{index}"

    col_hdr, col_adj = st.columns([3, 1])
    with col_hdr:
        status = "✅ Confirmed" if confirmed_flags[index] else "🔲 Pending"
        st.subheader(f"Image {index + 1} of {total} — {status}")
        st.progress(n_confirmed / total, text=f"{n_confirmed}/{total} confirmed")
    with col_adj:
        page_adjustment = int(
            st.number_input(
                "Page offset",
                step=1,
                help="Actual page = PDF page + offset",
                key="page_adjustment",
            )
        )
    doc_page = pdf_page + page_adjustment

    # -----------------------------------------------------------------------
    # Prefetch: move background result into session cache
    # -----------------------------------------------------------------------
    with _PREFETCH_LOCK:
        if index in _PREFETCH_CACHE:
            st.session_state.ocr_page_cache[index] = _PREFETCH_CACHE.pop(index)

    # If currently being computed in background, wait for it
    with _PREFETCH_LOCK:
        is_current_bg = index in _PREFETCH_IN_PROGRESS

    if is_current_bg and index not in st.session_state.ocr_page_cache:
        with st.spinner(f"Finishing background OCR on image {index + 1}…"):
            while True:
                with _PREFETCH_LOCK:
                    if index not in _PREFETCH_IN_PROGRESS:
                        if index in _PREFETCH_CACHE:
                            st.session_state.ocr_page_cache[index] = (
                                _PREFETCH_CACHE.pop(index)
                            )
                        break
                time.sleep(0.1)

    # Auto-trigger OCR if not cached
    if index not in st.session_state.ocr_page_cache:
        with st.spinner(f"Running Tesseract + VLM OCR on image {index + 1}…"):
            curr_input = image_paths[index] if image_paths else legacy_images
            page_data = _process_and_ocr_for_index(
                index=index,
                image_input=curr_input,
                pages=pages,
                pipeline_steps=preprocess_steps,
                page_adjustment=page_adjustment,
                psm_config=psm_config,
                min_width=min_width,
                min_height=min_height,
            )
            st.session_state.ocr_page_cache[index] = page_data
            st.session_state[editor_key] = page_data["baseline"]
            vlm_res = page_data.get("vlm_result")
            if vlm_res is not None and vlm_res.paragraphs:
                st.session_state[f"ocr_vlm_widget_{index}"] = "\n\n".join(
                    p.text for p in vlm_res.paragraphs
                )
        st.rerun()

    page_data = st.session_state.ocr_page_cache[index]
    tess_result: OCRPage | None = page_data.get("tess_result")
    tess_annotated = page_data.get("tess_annotated")
    tess_annotated_paras = page_data.get("tess_annotated_paras")
    tess_filtered_raw = page_data.get("tess_filtered_raw")
    tess_error: str | None = page_data.get("tess_error")
    vlm_result: OCRPage | None = page_data.get("vlm_result")
    vlm_error: str | None = page_data.get("vlm_error")

    vlm_key = f"ocr_vlm_widget_{index}"

    # Default raw VLM baseline text
    vlm_raw_baseline = ""
    if vlm_result is not None and vlm_result.paragraphs:
        vlm_raw_baseline = "\n\n".join(p.text for p in vlm_result.paragraphs)
    elif vlm_error:
        vlm_raw_baseline = f"Error running VLM: {vlm_error}"

    # Ensure editor and VLM widget keys exist
    if editor_key not in st.session_state:
        st.session_state[editor_key] = page_data.get("baseline", "")
    if vlm_key not in st.session_state:
        st.session_state[vlm_key] = vlm_raw_baseline

    # -----------------------------------------------------------------------
    # Background prefetch for next image
    # -----------------------------------------------------------------------
    next_index = index + 1
    if next_index < total and next_index not in st.session_state.ocr_page_cache:
        with _PREFETCH_LOCK:
            should_prefetch = (
                next_index not in _PREFETCH_IN_PROGRESS
                and next_index not in _PREFETCH_CACHE
            )
            if should_prefetch:
                _PREFETCH_IN_PROGRESS.add(next_index)

        if should_prefetch:
            next_input = image_paths[next_index] if image_paths else legacy_images

            def _bg_task(
                idx=next_index,
                img_inp=next_input,
                pgs=pages,
                steps=preprocess_steps,
                adj=page_adjustment,
                psm=psm_config,
                min_w=min_width,
                min_h=min_height,
            ) -> None:
                try:
                    res = _process_and_ocr_for_index(
                        index=idx,
                        image_input=img_inp,
                        pages=pgs,
                        pipeline_steps=steps,
                        page_adjustment=adj,
                        psm_config=psm,
                        min_width=min_w,
                        min_height=min_h,
                    )
                    with _PREFETCH_LOCK:
                        _PREFETCH_CACHE[idx] = res
                except Exception as exc:
                    with _PREFETCH_LOCK:
                        _PREFETCH_CACHE[idx] = {
                            "preprocessed_image": None,
                            "tess_result": None,
                            "tess_annotated": None,
                            "tess_error": str(exc),
                            "vlm_result": None,
                            "vlm_error": str(exc),
                            "baseline": "",
                        }
                finally:
                    with _PREFETCH_LOCK:
                        _PREFETCH_IN_PROGRESS.discard(idx)

            t = threading.Thread(target=_bg_task, daemon=True)
            t.start()

    # Prefetch status indicator (live auto-refreshing via fragment)
    if next_index < total:
        _render_prefetch_status(next_index, total)

    # -----------------------------------------------------------------------
    # Live warning computation
    # -----------------------------------------------------------------------
    current_text = st.session_state.get(editor_key, "")
    current_vlm_text = st.session_state.get(vlm_key, "")
    (
        warnings,
        hard_mismatch_indices_by_para,
        soft_mismatch_indices_by_para,
        missing_words_by_para,
        missing_paras_by_idx,
    ) = _compute_warnings(current_vlm_text, current_text)

    # -----------------------------------------------------------------------
    # Image navigation: select box & next prefetch
    # -----------------------------------------------------------------------
    def _format_image_option(i: int) -> str:
        flag = "✅" if confirmed_flags[i] else "🔲"
        p_num = pages[i] if i < len(pages) else i + 1
        name = (
            image_paths[i].name
            if (image_paths and i < len(image_paths))
            else f"page_{p_num:03d}"
        )
        return f"{flag} Image {i + 1}/{total} (PDF page {p_num}) — {name}"

    def _on_image_select() -> None:
        selected_i = st.session_state.get("image_selector_box")
        if isinstance(selected_i, int):
            target = selected_i
        elif isinstance(selected_i, str) and selected_i.isdigit():
            target = int(selected_i)
        else:
            target = None

        if target is not None and 0 <= target < total:
            st.session_state.ocr_current_index = target
            st.session_state["image_selector_box"] = target

    def _go_prev() -> None:
        if st.session_state.ocr_current_index > 0:
            st.session_state.ocr_current_index -= 1
            st.session_state["image_selector_box"] = st.session_state.ocr_current_index

    def _go_next() -> None:
        if st.session_state.ocr_current_index < total - 1:
            st.session_state.ocr_current_index += 1
            st.session_state["image_selector_box"] = st.session_state.ocr_current_index

    # Always ensure image_selector_box in session_state is an int matching current index
    raw_box_val = st.session_state.get("image_selector_box")
    if not isinstance(raw_box_val, int) or raw_box_val != index:
        st.session_state["image_selector_box"] = index

    sel_col1, sel_col2, sel_col3 = st.columns([1, 4, 1])
    with sel_col1:
        st.button(
            "← Previous Image",
            disabled=(index == 0),
            key="btn_prev_image_top",
            on_click=_go_prev,
        )
    with sel_col2:
        st.selectbox(
            "Select Image to Review",
            options=list(range(total)),
            index=index,
            format_func=_format_image_option,
            key="image_selector_box",
            on_change=_on_image_select,
            label_visibility="collapsed",
        )
    with sel_col3:
        st.button(
            "Next Image →",
            disabled=(index >= total - 1),
            key="btn_next_image_top",
            on_click=_go_next,
        )

    # -----------------------------------------------------------------------
    # Three-column review area
    # -----------------------------------------------------------------------
    bbox_session_key = f"ocr_bbox_list_{index}"
    current_reviewed_paragraphs = _split_paragraphs(current_text)

    # Initialize or synchronize bounding boxes with current reviewed paragraphs
    prev_bbox_list = st.session_state.get(bbox_session_key, [])
    synced_bbox_rows = _sync_paragraph_bboxes(
        prev_bbox_list,
        current_reviewed_paragraphs,
        tess_page=tess_result,
        filtered_raw=tess_filtered_raw,
    )
    st.session_state[bbox_session_key] = synced_bbox_rows

    col_img, col_edit, col_ref = st.columns([1.5, 1, 1])

    with col_img:
        col_mode, col_rerun = st.columns([1, 2])

        with col_mode:
            display_options = [
                "Paragraphs (Annotated)",
                "Words (Annotated)",
                "Original image",
            ]
            if is_modified:
                display_options.insert(2, "Preprocessed image")
            saved_mode = st.session_state.get(
                f"image_view_mode_{index}", "Paragraphs (Annotated)"
            )
            mode_idx = (
                display_options.index(saved_mode)
                if saved_mode in display_options
                else 0
            )
            view_mode = st.radio(
                "Image display mode",
                display_options,
                index=mode_idx,
                horizontal=True,
                key=f"image_view_mode_{index}",
            )

        with col_rerun:
            st.subheader(f"PDF page {pdf_page} (actual page {doc_page})")
            if st.button("Re-run OCR on current image"):
                st.session_state.ocr_page_cache.pop(index, None)
                with _PREFETCH_LOCK:
                    _PREFETCH_CACHE.pop(index, None)
                st.session_state.pop(editor_key, None)
                st.session_state.pop(f"ocr_vlm_widget_{index}", None)
                st.session_state.pop(bbox_session_key, None)
                for k in list(st.session_state.keys()):
                    if k.startswith(f"ocr_bbox_ctrl_{index}_"):
                        del st.session_state[k]
                st.rerun()

        # Load active image on-demand for display
        current_image = (
            _load_image_from_disk(image_paths[index])
            if image_paths
            else legacy_images[index]
        )

        # Choose what to display
        preprocessed_image = page_data.get("preprocessed_image")
        base_img = (
            preprocessed_image if preprocessed_image is not None else current_image
        )
        img_width, img_height = _image_to_pil(base_img).size
        if view_mode == "Words (Annotated)":
            if tess_error:
                st.error(f"Tesseract failed: {tess_error}")
            elif tess_annotated is not None:
                st.image(tess_annotated, width="stretch")
            elif preprocessed_image is not None:
                st.image(_image_to_pil(preprocessed_image), width="stretch")
            else:
                st.image(_image_to_pil(current_image), width="stretch")
        elif view_mode == "Paragraphs (Annotated)":
            if tess_error:
                st.error(f"Tesseract failed: {tess_error}")
            else:
                # Live dynamic drawing using currently adjusted bounding boxes
                base_img = (
                    preprocessed_image
                    if preprocessed_image is not None
                    else current_image
                )
                current_live_paras = [
                    OCRParagraph(
                        paragraph_id=i + 1,
                        bbox=(
                            int(r["x_min"]),
                            int(r["y_min"]),
                            int(r["x_max"]),
                            int(r["y_max"]),
                        )
                        if (
                            r.get("x_min") is not None
                            and r.get("y_min") is not None
                            and r.get("x_max") is not None
                            and r.get("y_max") is not None
                            and not (
                                r.get("x_min") == 0
                                and r.get("y_min") == 0
                                and r.get("x_max") == 0
                                and r.get("y_max") == 0
                            )
                        )
                        else None,
                        text=r.get("Preview", ""),
                    )
                    for i, r in enumerate(synced_bbox_rows)
                ]
                live_annotated_paras = _draw_paragraph_bboxes(
                    base_img, current_live_paras
                )
                st.image(live_annotated_paras, width="stretch")
        elif view_mode == "Preprocessed image":
            if preprocessed_image is not None:
                st.image(
                    _image_to_pil(preprocessed_image),
                    caption="Preprocessed image (Tesseract input)",
                    width="stretch",
                )
            else:
                st.image(_image_to_pil(current_image), width="stretch")
        else:
            st.image(
                _image_to_pil(current_image),
                caption="Original image (VLM input)",
                width="stretch",
            )
        if vlm_error:
            st.error(f"VLM failed: {vlm_error}")

    with col_edit:
        st.badge("Separate paragraphs with a blank line.")

        st.subheader("Final text")
        editable_html = _render_editable_highlighted_html(
            current_text,
            hard_mismatch_indices_by_para,
            soft_mismatch_indices_by_para=soft_mismatch_indices_by_para,
        )
        if _EDITABLE_HTML_BOX is not None:
            editable_result = _EDITABLE_HTML_BOX(
                key=f"{editor_key}_html",
                data={
                    "html": editable_html,
                    "plain": "\n\n".join(_split_paragraphs(current_text)),
                },
                on_text_change=lambda: None,
            )
            edited_text = getattr(editable_result, "text", None)
            if edited_text is not None and edited_text != current_text:
                st.session_state[editor_key] = edited_text
                st.rerun()
        else:
            st.text_area(
                "Reviewed text",
                height=250,
                key=editor_key,
                label_visibility="collapsed",
            )

        st.subheader("Paragraph Bounding Boxes")
        st.caption("Adjust (x_min, y_min, x_max, y_max) for each paragraph.")
        _render_paragraph_bbox_controls(
            index=index,
            synced_bbox_rows=synced_bbox_rows,
            bbox_session_key=bbox_session_key,
            img_width=img_width,
            img_height=img_height,
        )

        confirm_label = (
            "✅ Confirm & Save Page"
            if not confirmed_flags[index]
            else "🔄 Re-confirm Page"
        )
        if st.button(confirm_label, type="primary", width="stretch"):
            final_text = st.session_state.get(editor_key, "")
            # Extract customized bboxes from session state
            curr_bbox_rows = st.session_state.get(bbox_session_key, [])
            custom_boxes: list[tuple[int, int, int, int] | None] = []
            for r in curr_bbox_rows:
                if (
                    r.get("x_min") is not None
                    and r.get("y_min") is not None
                    and r.get("x_max") is not None
                    and r.get("y_max") is not None
                    and not (
                        r.get("x_min") == 0
                        and r.get("y_min") == 0
                        and r.get("x_max") == 0
                        and r.get("y_max") == 0
                    )
                ):
                    custom_boxes.append(
                        (
                            int(r["x_min"]),
                            int(r["y_min"]),
                            int(r["x_max"]),
                            int(r["y_max"]),
                        )
                    )
                else:
                    custom_boxes.append(None)

            page_obj = _build_ocr_page(
                text=final_text,
                pdf_page_number=pdf_page,
                document_page_number=doc_page,
                tess_page=tess_result,
                filtered_raw=tess_filtered_raw,
                custom_bboxes=custom_boxes if custom_boxes else None,
            )

            # 2-step validation: check parsed document page number against calculated doc_page
            parsed_doc_page, _ = (None, None)
            if page_obj.paragraphs:
                first_para_text = page_obj.paragraphs[0].text
                parsed_doc_page, _ = parse_header_paragraph(
                    first_para_text,
                    language=st.session_state.get("document_language", "vi"),
                )

            if parsed_doc_page is not None and parsed_doc_page != doc_page:
                mismatch_payload = {
                    "calculated_page": doc_page,
                    "parsed_page": parsed_doc_page,
                    "pending_data": page_obj.model_dump(),
                    "index": index,
                    "total": total,
                }
                st.session_state["pending_page_mismatch"] = mismatch_payload
                if hasattr(st, "dialog"):
                    st.session_state["show_page_mismatch_dialog"] = True
                    st.rerun()
                else:
                    st.session_state["show_inline_page_mismatch_warning"] = True
                    st.rerun()
            else:
                _execute_page_save(page_obj.model_dump(), output_path, index, total)

    with col_ref:
        highlighted_html = _render_highlighted_text(
            current_text,
            hard_mismatch_indices_by_para,
            soft_mismatch_indices_by_para=soft_mismatch_indices_by_para,
            missing_words_by_para=missing_words_by_para,
            missing_paras_by_idx=missing_paras_by_idx,
        )
        with st.container(height=200, border=True):
            st.markdown(highlighted_html, unsafe_allow_html=True)

        if warnings:
            icons = {"mismatch": "⚡", "soft_mismatch": "🔶", "orphan": "🔀"}
            st.write(f"⚠️ {len(warnings)} Warning(s) & Discrepancies")
            with st.container(height=200, border=True):
                for w in warnings:
                    icon = icons.get(w["type"], "⚠️")
                    st.markdown(f"{icon} **{w['label']}** - {w['msg']}")
        else:
            st.success("✅ No mismatches detected.")

        st.subheader("VLM Reference Text")
        st.text_area(
            "VLM Reference Text",
            height=250,
            key=vlm_key,
            label_visibility="collapsed",
        )

        if current_vlm_text != vlm_raw_baseline and vlm_raw_baseline:
            if st.button("Reset VLM text", key=f"reset_vlm_{index}"):
                st.session_state[vlm_key] = vlm_raw_baseline
                st.rerun()

    # -----------------------------------------------------------------------
    # Finalize section — always visible
    # -----------------------------------------------------------------------
    st.divider()
    all_confirmed = all(confirmed_flags[:total])

    if all_confirmed:
        st.subheader("Finalize — Export")
        st.success(
            f"All {total} image(s) confirmed. Ready to write the final document."
        )
    else:
        st.subheader(f"Save Progress ({n_confirmed}/{total} images confirmed)")
        st.info(
            f"You have confirmed {n_confirmed} of {total} images. "
            "You can save intermediate progress or export PPOCRLabel files anytime."
        )

    source_name = st.session_state.get("document_name", "document.pdf")
    st.caption(f"Source document: **{source_name}**")

    # ---- Intermediate save (pages JSON) ----
    save_btn_label = (
        "💾 Save Progress (pages JSON)"
        if not all_confirmed
        else f"💾 Save Progress as pages JSON ({n_confirmed} pages)"
    )

    if st.button(save_btn_label, disabled=(n_confirmed == 0)):
        _handle_save_request(
            output_path=output_path,
            source_name=source_name,
            language=language,
            sample_image_path=sample_image_path,
            images=legacy_images if not image_paths else None,
        )

    # Provide download button if output file currently exists
    out_file_path = Path(output_path)
    if out_file_path.exists():
        try:
            doc_content = out_file_path.read_text(encoding="utf-8")
            st.download_button(
                "⬇️ Download Current Pages JSON",
                data=doc_content,
                file_name=out_file_path.name,
                mime="application/json",
            )
        except Exception:
            pass

    st.divider()

    # ---- PPOCRLabel export ----
    st.subheader("Export PPOCRLabel")
    export_dir_input = st.text_input(
        "Export directory",
        value=st.session_state.get("ppocr_export_dir", default_export_dir),
        key="ppocr_export_dir",
        help="Directory where Label.txt and fileState.txt will be written.",
    )

    if st.button(
        f"📤 Export PPOCRLabel ({n_confirmed} pages)",
        type="primary",
        disabled=(n_confirmed == 0),
    ):
        _handle_export_request(
            export_dir_input, source_name, language, image_paths=image_paths
        )


if __name__ == "__main__":
    main()
