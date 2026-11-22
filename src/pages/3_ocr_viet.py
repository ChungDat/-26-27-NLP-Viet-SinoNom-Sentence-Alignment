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
        OCRDocument,
        OCRPage,
        OCRParagraph,
        OCRWord,
        TesseractOCR,
    )
except ImportError:
    from ocr_viet import (  # type: ignore[no-redef]
        VLMOCR,
        OCRDocument,
        OCRPage,
        OCRParagraph,
        OCRWord,
        TesseractOCR,
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
    """Split text into paragraphs on blank lines; strip empty results.

    Args:
        text: Multi-paragraph text with blank-line paragraph separators.

    Returns:
        List of non-empty paragraph strings.
    """
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _build_ocr_page(
    text: str,
    pdf_page_number: int,
    document_page_number: int,
) -> OCRPage:
    """Build a reviewed OCRPage from plain editor text.

    Args:
        text: Editor text with blank-line paragraph separators.
        pdf_page_number: 1-indexed PDF reader page number.
        document_page_number: Actual document page number after offset.

    Returns:
        An OCRPage with method="reviewed".
    """
    paragraphs: list[OCRParagraph] = []
    for idx, para_text in enumerate(_split_paragraphs(text), start=1):
        words = [OCRWord(text=w, bbox=None, confidence=None) for w in para_text.split()]
        paragraphs.append(
            OCRParagraph(
                paragraph_id=idx,
                bbox=None,
                text=" ".join(w.text for w in words),
                words=words,
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
            ("ocr_editor_widget_", "ocr_vlm_widget_", "image_view_mode_")
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
            words = para.split()
            hard_flagged = hard_mismatch_indices_by_para.get(i, set())
            soft_flagged = soft_mismatch_indices_by_para.get(i, set())
            para_missing = missing_words_by_para.get(i, {})
            formatted_words: list[str] = []

            for w_idx, w in enumerate(words):
                # Render any missing insertions before this word
                if w_idx in para_missing:
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

            # Render missing insertions at end of paragraph
            if len(words) in para_missing:
                for missing_seg in para_missing[len(words)]:
                    escaped_seg = html.escape(missing_seg)
                    formatted_words.append(
                        f"<span style='color: red; font-weight: bold; "
                        f"text-decoration: underline;'>[MISSING: {escaped_seg}]</span>"
                    )

            para_html = " ".join(formatted_words)
            html_paragraphs.append(
                f"<p style='margin: 0 0 10px 0; line-height: 1.5;'>{para_html}</p>"
            )

    return "".join(html_paragraphs)


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
    """Load confirmed pages from the staging JSONL file.

    Args:
        output_path: Path to the final OCRDocument JSON (determines staging path).

    Returns:
        List of OCRPage objects deserialized from the staging file.
    """
    staging = _staging_path(output_path)
    if not staging.exists():
        return []
    pages: list[OCRPage] = []
    for line in staging.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                pages.append(OCRPage.model_validate_json(line))
            except Exception:
                pass
    return pages


# ---------------------------------------------------------------------------
# Document saving helpers
# ---------------------------------------------------------------------------


def _perform_save_document(
    output_path: str,
    source_name: str,
    language: str,
    images: list[np.ndarray],
) -> tuple[int, Path]:
    """Write confirmed pages to an OCRDocument JSON file.

    Preprocessed images are saved at confirm time (not here). The staging file
    is also synced after writing.

    Args:
        output_path: Destination path for the OCRDocument JSON.
        source_name: Source PDF filename recorded in the document.
        language: Document language code (e.g. "vie").
        images: Original extracted images (used only to read dimensions).

    Returns:
        A 2-tuple of (n_saved_pages, out_path).
    """
    confirmed_pages_raw = st.session_state.get("ocr_confirmed_pages", [])
    confirmed_page_objs = [OCRPage(**p) for p in confirmed_pages_raw if p is not None]
    h, w = images[0].shape[:2] if images else (None, None)
    doc = OCRDocument(
        source=source_name,
        language=language,
        page_count=len(confirmed_page_objs),
        width=int(w) if w is not None else None,
        height=int(h) if h is not None else None,
        pages=confirmed_page_objs,
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")

    # Sync staging file
    _sync_staging_file(confirmed_pages_raw, output_path)

    return len(confirmed_page_objs), out_path


# Overwrite confirmation dialog (requires Streamlit >= 1.29 for st.dialog)
if hasattr(st, "dialog"):

    @st.dialog("⚠️ File Already Exists")
    def _confirm_overwrite_dialog(
        output_path: str,
        source_name: str,
        language: str,
        images: list[np.ndarray],
    ) -> None:
        """Modal dialog to confirm overwriting an existing OCRDocument JSON.

        Args:
            output_path: Path to the existing file.
            source_name: Source PDF filename.
            language: Document language code.
            images: Original extracted images.
        """
        st.warning(f"File `{output_path}` already exists.")
        st.write("Do you want to overwrite the existing file?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, Overwrite", type="primary", use_container_width=True):
                n_pages, p = _perform_save_document(
                    output_path, source_name, language, images
                )
                msg = f"Saved {n_pages} page(s) to `{p}`."
                st.session_state["save_alert"] = ("success", msg)
                st.rerun()
        with col2:
            if st.button("Cancel", use_container_width=True):
                st.rerun()


def _handle_save_request(
    output_path: str,
    source_name: str,
    language: str,
    images: list[np.ndarray],
) -> None:
    """Handle a save request, checking for existing files and prompting overwrite.

    Args:
        output_path: Destination path for the OCRDocument JSON.
        source_name: Source PDF filename.
        language: Document language code.
        images: Original extracted images.
    """
    out_p = Path(output_path)
    if out_p.exists() and not st.session_state.get("overwrite_approved", False):
        if hasattr(st, "dialog"):
            _confirm_overwrite_dialog(output_path, source_name, language, images)
        else:
            # Fallback for environments where st.dialog is unavailable
            st.session_state["show_inline_overwrite_warning"] = True
            st.rerun()
    else:
        st.session_state["overwrite_approved"] = False
        n_pages, p = _perform_save_document(output_path, source_name, language, images)
        msg = f"Saved {n_pages} page(s) to `{p}`."
        st.session_state["save_alert"] = ("success", msg)
        st.rerun()


# ---------------------------------------------------------------------------
# OCR execution
# ---------------------------------------------------------------------------


def _process_and_ocr_for_index(
    index: int,
    raw_images: list[np.ndarray],
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
        index: Index into raw_images and pages lists.
        raw_images: List of original (un-preprocessed) page images.
        pages: List of PDF page numbers corresponding to raw_images.
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

    # Parallel execution: run (Preprocessing + Tesseract) and (VLM OCR) concurrently
    def _prep_and_tess() -> tuple[
        np.ndarray, OCRPage | None, Image.Image | None, str | None
    ]:
        try:
            prep = PreprocessPipeline(pipeline_steps).apply(raw_images[index])
        except Exception:
            prep = raw_images[index]

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
            annotated = tess.annotate_image()
            return prep, res, annotated, None
        except Exception as e:
            return prep, None, None, str(e)

    def _run_vlm() -> tuple[OCRPage | None, str | None]:
        try:
            vlm = VLMOCR(model=VLM_MODEL)
            res = vlm.run(
                raw_images[index],
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

        preprocessed_image, tess_result, tess_annotated, tess_error = (
            tess_future.result()
        )
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
        "⚙️ OCR Settings",
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
        with cfg_col2:
            min_width = int(
                st.number_input(
                    "Min bbox width (px)",
                    min_value=0,
                    value=50,
                    step=1,
                    key="bbox_min_width",
                )
            )
            min_height = int(
                st.number_input(
                    "Min bbox height (px)",
                    min_value=0,
                    value=50,
                    step=1,
                    key="bbox_min_height",
                )
            )
        with cfg_col3:
            output_path = st.text_input(
                "Output JSON path",
                value=st.session_state.get("ocr_output_path", "output/ocr_result.json"),
                key="ocr_output_path",
            )
            st.text_input(
                "Source name",
                value=st.session_state.get("pdf_filename", ""),
                key="ocr_source_name",
            )

        is_modified = st.session_state.get("preprocess_modified", bool(preprocess_steps))
        if is_modified:
            st.caption(
                f"ℹ️ Preprocessing pipeline: **Active ({len(preprocess_steps)} step{'s' if len(preprocess_steps) != 1 else ''})**"
            )
        else:
            st.caption("ℹ️ Preprocessing pipeline: **None (using raw images)**")

    # -----------------------------------------------------------------------
    # Image lists
    # -----------------------------------------------------------------------
    images: list[np.ndarray] = st.session_state.get("images", [])
    pages: list[int] = st.session_state.get("pages", [])

    if not images:
        st.info("Extract PDF pages on the first page before running OCR.")
        return

    total = len(images)

    # Reset cache if image count changed
    if (
        "ocr_images_signature" not in st.session_state
        or st.session_state.ocr_images_signature != total
    ):
        st.session_state.ocr_images_signature = total
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
    # Sidebar: Progress & Save
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.divider()
        st.subheader("Progress & Save")
        st.caption(f"{n_confirmed} of {total} page(s) confirmed.")
        if st.button(
            f"💾 Save Progress ({n_confirmed}/{total})",
            disabled=(n_confirmed == 0),
            use_container_width=True,
            help="Save confirmed pages to OCRDocument JSON immediately.",
        ):
            source_name = st.session_state.get(
                "ocr_source_name", st.session_state.get("pdf_filename", "document.pdf")
            )
            language = st.session_state.get("document_language", "vi")
            _handle_save_request(
                output_path=output_path,
                source_name=source_name,
                language=language,
                images=images,
            )

    # -----------------------------------------------------------------------
    # Save notification banners
    # -----------------------------------------------------------------------
    if "save_alert" in st.session_state:
        level, msg = st.session_state.pop("save_alert")
        if level == "success":
            st.success(msg)
        elif level == "error":
            st.error(msg)

    # Fallback inline overwrite warning for older Streamlit
    if st.session_state.get("show_inline_overwrite_warning", False):
        st.warning(f"File `{output_path}` already exists.")
        col_ow1, col_ow2 = st.columns(2)
        with col_ow1:
            if st.button("⚠️ Confirm Overwrite File", type="primary"):
                st.session_state["show_inline_overwrite_warning"] = False
                st.session_state["overwrite_approved"] = True
                source_name = st.session_state.get(
                    "ocr_source_name",
                    st.session_state.get("pdf_filename", "document.pdf"),
                )
                language = st.session_state.get("document_language", "vi")
                _handle_save_request(
                    output_path=output_path,
                    source_name=source_name,
                    language=language,
                    images=images,
                )
        with col_ow2:
            if st.button("Cancel"):
                st.session_state["show_inline_overwrite_warning"] = False
                st.rerun()

    # -----------------------------------------------------------------------
    # Header: page info & navigation controls
    # -----------------------------------------------------------------------
    pdf_page = pages[index]
    page_adjustment = int(st.session_state.get("page_adjustment", 0))
    doc_page = pdf_page + page_adjustment
    editor_key = f"ocr_editor_widget_{index}"

    col_hdr, col_adj = st.columns([3, 1])
    with col_hdr:
        status = "✅ Confirmed" if confirmed_flags[index] else "🔲 Pending"
        st.subheader(
            f"Image {index + 1} of {total} — PDF page {pdf_page} (actual page {doc_page}) — {status}"
        )
        st.progress(n_confirmed / total, text=f"{n_confirmed}/{total} confirmed")
    with col_adj:
        page_adjustment = st.number_input(
            "Page offset",
            value=int(st.session_state.get("page_adjustment", 0)),
            step=1,
            help="Actual page = PDF page + offset",
            key="page_adjustment",
        )

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
            page_data = _process_and_ocr_for_index(
                index=index,
                raw_images=images,
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

            def _bg_task(
                idx=next_index,
                imgs=images,
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
                        raw_images=imgs,
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
    # Three-column review area
    # -----------------------------------------------------------------------
    col_img, col_warn, col_text = st.columns([1.5, 1, 1])

    with col_img:
        col_mode, col_rerun = st.columns([1, 2])

        with col_mode:
            display_options = ["Annotated", "Original image"]
            if is_modified:
                display_options.insert(1, "Preprocessed image")
            saved_mode = st.session_state.get(f"image_view_mode_{index}", "Annotated")
            mode_idx = display_options.index(saved_mode) if saved_mode in display_options else 0
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
                st.rerun()

        # Choose what to display
        preprocessed_image = page_data.get("preprocessed_image")
        if view_mode == "Annotated":
            if tess_error:
                st.error(f"Tesseract failed: {tess_error}")
            elif tess_annotated is not None:
                st.image(tess_annotated, width="stretch")
            elif preprocessed_image is not None:
                st.image(_image_to_pil(preprocessed_image), width="stretch")
            else:
                st.image(_image_to_pil(images[index]), width="stretch")
        elif view_mode == "Preprocessed image":
            if preprocessed_image is not None:
                st.image(
                    _image_to_pil(preprocessed_image),
                    caption="Preprocessed image (Tesseract input)",
                    width="stretch",
                )
            else:
                st.image(_image_to_pil(images[index]), width="stretch")
        else:
            st.image(
                _image_to_pil(images[index]),
                caption="Original image (VLM input)",
                width="stretch",
            )
        if vlm_error:
            st.error(f"VLM failed: {vlm_error}")

    with col_warn:
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
            with st.container(height=300, border=True):
                for w in warnings:
                    icon = icons.get(w["type"], "⚠️")
                    st.markdown(f"{icon} **{w['label']}** - {w['msg']}")
        else:
            st.success("✅ No mismatches detected.")

    with col_text:
        st.badge("Separate paragraphs with a blank line.")

        st.subheader("Final text")
        st.text_area(
            "Reviewed text",
            height=250,
            key=editor_key,
            label_visibility="collapsed",
        )

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

        # Navigation & confirm
        nav_prev, nav_confirm, nav_next = st.columns([1, 2, 1])

        with nav_prev:
            if st.button("← Previous", disabled=(index == 0)):
                st.session_state.ocr_current_index = index - 1
                st.rerun()

        with nav_confirm:
            confirm_label = (
                "✅ Confirm & Save Page"
                if not confirmed_flags[index]
                else "🔄 Re-confirm Page"
            )
            if st.button(confirm_label, type="primary"):
                final_text = st.session_state.get(editor_key, "")
                page_obj = _build_ocr_page(
                    text=final_text,
                    pdf_page_number=pdf_page,
                    document_page_number=doc_page,
                )
                confirmed_flags[index] = True
                st.session_state.ocr_confirmed_flags = confirmed_flags
                st.session_state.ocr_confirmed_pages[index] = page_obj.model_dump()

                # Save preprocessed image to disk immediately at confirm time
                result = st.session_state.ocr_page_cache.get(index, {})
                prep_img = result.get("preprocessed_image")
                if prep_img is not None:
                    try:
                        language = st.session_state.get("document_language", "vi")
                        prep_dir = Path(output_path).parent / language
                        prep_dir.mkdir(parents=True, exist_ok=True)
                        img_path = prep_dir / f"{source_name}.{pdf_page:03d}.jpg"
                        _image_to_pil(prep_img).save(img_path)
                    except Exception as img_err:
                        st.warning(f"Could not save preprocessed image: {img_err}")

                # Rewrite staging file with all currently confirmed pages
                try:
                    _sync_staging_file(
                        st.session_state.ocr_confirmed_pages, output_path
                    )
                    st.success(f"Page {index + 1} saved to staging file.")
                except Exception as e:
                    st.error(f"Could not write staging file: {e}")

                # Auto-advance if not last
                if index < total - 1:
                    st.session_state.ocr_current_index = index + 1
                    st.rerun()
                else:
                    st.rerun()

        with nav_next:
            next_disabled = not confirmed_flags[index] or index >= total - 1
            if st.button("Next →", disabled=next_disabled):
                st.session_state.ocr_current_index = index + 1
                st.rerun()

    # -----------------------------------------------------------------------
    # Finalize section — always visible
    # -----------------------------------------------------------------------
    st.divider()
    all_confirmed = all(confirmed_flags[:total])

    if all_confirmed:
        st.subheader("Finalize - Write OCRDocument")
        st.success(
            f"All {total} image(s) confirmed. Ready to write the final document."
        )
    else:
        st.subheader(f"Save Progress ({n_confirmed}/{total} images confirmed)")
        st.info(
            f"You have confirmed {n_confirmed} of {total} images. "
            "You can write an intermediate OCRDocument JSON anytime."
        )

    source_name = st.text_input(
        "Source (PDF filename)",
        value=st.session_state.get(
            "ocr_source_name", st.session_state.get("pdf_filename", "")
        ),
        key="ocr_finalize_source_name",
    )

    save_btn_label = (
        "Write OCRDocument JSON"
        if all_confirmed
        else f"💾 Save Progress as OCRDocument ({n_confirmed} pages)"
    )

    if st.button(save_btn_label, type="primary", disabled=(n_confirmed == 0)):
        _handle_save_request(
            output_path=output_path,
            source_name=source_name,
            language=language,
            images=images,
        )

    # Provide download button if output file currently exists
    out_file_path = Path(output_path)
    if out_file_path.exists():
        try:
            doc_content = out_file_path.read_text(encoding="utf-8")
            st.download_button(
                "Download Current OCRDocument JSON",
                data=doc_content,
                file_name=out_file_path.name,
                mime="application/json",
            )
        except Exception:
            pass


if __name__ == "__main__":
    main()
