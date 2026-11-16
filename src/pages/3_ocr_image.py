from __future__ import annotations

import html
import json
import re
from itertools import zip_longest
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

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
    from ocr_viet import (
        VLMOCR,
        OCRDocument,
        OCRPage,
        OCRParagraph,
        OCRWord,
        TesseractOCR,
    )

LOW_CONFIDENCE_THRESHOLD = 85.0
VLM_MODEL = "qwen3-vl:2b-instruct"

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
    if image.ndim == 2:
        return Image.fromarray(image)
    if image.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGBA2RGB))
    return Image.fromarray(image)


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on blank lines; strip empty results."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _build_ocr_page(
    text: str,
    pdf_page_number: int,
    document_page_number: int,
) -> OCRPage:
    """Build a reviewed OCRPage from plain editor text."""
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
        page_name=f"page_{pdf_page_number:04d}",
        method="reviewed",
        paragraphs=paragraphs,
    )


# warning computation


def _align_paragraph_words(
    editor_words: list[str],
    vlm_words: list[str],
    max_lookahead: int = 5,
) -> tuple[list[dict], set[int]]:
    """Realign word sequences using 2-pointer with bounded lookahead to prevent cascading warnings.

    Returns:
        (discrepancies, editor_mismatch_indices)
        where editor_mismatch_indices is the set of word indices in editor_words that have discrepancies.
    """
    discrepancies: list[dict] = []
    editor_mismatch_indices: set[int] = set()
    i = 0
    j = 0
    m = len(editor_words)
    n = len(vlm_words)

    while i < m and j < n:
        if editor_words[i] == vlm_words[j]:
            i += 1
            j += 1
            continue

        # Search for the nearest matching anchor within lookahead window K
        best_anchor: tuple[int, int] | None = None
        # Explore anchors by increasing Manhattan distance (di + dj)
        for total_dist in range(1, 2 * max_lookahead + 1):
            for di in range(min(total_dist, max_lookahead) + 1):
                dj = total_dist - di
                if dj <= max_lookahead and (i + di) < m and (j + dj) < n:
                    if editor_words[i + di] == vlm_words[j + dj]:
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
            elif di > 0 and dj == 0:
                msg = f'Extra word(s) in editor: "{seg_a}" (not in VLM)'
            else:
                msg = f'Current: "{seg_a}" | VLM: "{seg_b}"'

            for idx in range(i, i + di):
                editor_mismatch_indices.add(idx)

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
            # Fallback: advance both pointers by 1
            seg_a = editor_words[i]
            seg_b = vlm_words[j]
            editor_mismatch_indices.add(i)
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

    # Trailing words in editor
    if i < m:
        for idx in range(i, m):
            editor_mismatch_indices.add(idx)
        seg_a = " ".join(editor_words[i:])
        discrepancies.append(
            {
                "type": "mismatch",
                "editor_segment": seg_a,
                "vlm_segment": "",
                "msg": f'Extra word(s) in editor: "{seg_a}" (not in VLM)',
            }
        )
    # Trailing words in VLM
    elif j < n:
        seg_b = " ".join(vlm_words[j:])
        discrepancies.append(
            {
                "type": "mismatch",
                "editor_segment": "",
                "vlm_segment": seg_b,
                "msg": f'Missing word(s) in editor (VLM has: "{seg_b}")',
            }
        )

    return discrepancies, editor_mismatch_indices


def _compute_warnings(
    vlm_source: OCRPage | str | None,
    editor_text: str,
    max_lookahead: int = 5,
) -> tuple[list[dict], dict[int, set[int]]]:
    """Return a tuple of (warnings, mismatch_indices_by_para)."""
    warnings: list[dict] = []
    mismatch_indices_by_para: dict[int, set[int]] = {}

    editor_paras = _split_paragraphs(editor_text)

    if isinstance(vlm_source, str):
        vlm_paras_text = _split_paragraphs(vlm_source)
    elif vlm_source is not None:
        vlm_paras_text = [p.text for p in vlm_source.paragraphs]
    else:
        vlm_paras_text = []

    # Compare editor text against VLM text per-paragraph
    for i, (ep_text, vp_text) in enumerate(zip_longest(editor_paras, vlm_paras_text)):
        para_label = f"Paragraph {i + 1}"

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
            continue

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
            # All words in extra editor paragraph are flagged
            mismatch_indices_by_para[i] = set(range(len(ep_text.split())))
            continue

        if ep_text is not None and vp_text is not None:
            editor_words = ep_text.split()
            vlm_words = vp_text.split()

            align_discrepancies, para_mismatch_indices = _align_paragraph_words(
                editor_words, vlm_words, max_lookahead=max_lookahead
            )
            if para_mismatch_indices:
                mismatch_indices_by_para[i] = para_mismatch_indices

            for d in align_discrepancies:
                warnings.append(
                    {
                        "type": "mismatch",
                        "para": i,
                        "label": para_label,
                        "msg": d["msg"],
                    }
                )

    return warnings, mismatch_indices_by_para


def _render_highlighted_text(
    editor_text: str,
    mismatch_indices_by_para: dict[int, set[int]],
) -> str:
    """Build an HTML preview with warning words colored in pure red."""
    if not editor_text.strip():
        return "<p style='color: #888; font-style: italic; margin: 0;'>No text to preview.</p>"

    editor_paras = _split_paragraphs(editor_text)
    html_paragraphs: list[str] = []

    for i, para in enumerate(editor_paras):
        words = para.split()
        flagged_indices = mismatch_indices_by_para.get(i, set())
        formatted_words: list[str] = []

        for w_idx, w in enumerate(words):
            escaped_word = html.escape(w)
            if w_idx in flagged_indices:
                formatted_words.append(
                    f"<span style='color: red; font-weight: bold;'>{escaped_word}</span>"
                )
            else:
                formatted_words.append(escaped_word)

        para_html = " ".join(formatted_words)
        html_paragraphs.append(
            f"<p style='margin: 0 0 10px 0; line-height: 1.5;'>{para_html}</p>"
        )

    return "".join(html_paragraphs)


# staging file helpers


def _staging_path(output_path: str) -> Path:
    p = Path(output_path)
    return p.parent / (p.stem + "_staging.jsonl")


def _sync_staging_file(confirmed_pages: list[dict | None], output_path: str) -> None:
    """Rewrite staging file with all confirmed pages in JSON Lines format."""
    staging = _staging_path(output_path)
    staging.parent.mkdir(parents=True, exist_ok=True)
    with staging.open("w", encoding="utf-8") as f:
        for page_dict in confirmed_pages:
            if page_dict is not None:
                f.write(json.dumps(page_dict, ensure_ascii=False) + "\n")


def _load_staging(output_path: str) -> list[OCRPage]:
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


# sidebar


def _sidebar_controls() -> tuple[str, int, int, str]:
    """Render sidebar and return (psm_config, min_w, min_h, output_path)."""
    with st.sidebar:
        st.subheader("OCR settings")
        psm_config = st.selectbox(
            "Page segmentation method",
            list(PSM_NOTES),
            format_func=lambda c: f"{c} - {PSM_NOTES[c]}",
            key="psm_config",
        )
        st.subheader("Bounding-box filter")
        min_width = int(
            st.number_input(
                "Min box width (px)",
                min_value=0,
                value=50,
                step=1,
                key="bbox_min_width",
            )
        )
        min_height = int(
            st.number_input(
                "Min box height (px)",
                min_value=0,
                value=50,
                step=1,
                key="bbox_min_height",
            )
        )
        st.subheader("Output")
        output_path = st.text_input(
            "Output JSON path",
            value=st.session_state.get("ocr_output_path", "output/ocr_result.json"),
            key="ocr_output_path",
        )
    return psm_config, min_width, min_height, output_path


# OCR execution


def _run_ocr_for_index(
    index: int,
    images: list[np.ndarray],
    ocr_images: list[np.ndarray],
    pages: list[int],
    page_adjustment: int,
    psm_config: str,
    min_width: int,
    min_height: int,
) -> dict:
    """Run both engines for image at *index* and return the result dict."""
    pdf_page = pages[index]
    doc_page = pdf_page + page_adjustment
    page_name = f"page_{pdf_page:04d}"

    tess_result = None
    tess_annotated = None
    tess_error = None
    vlm_result = None
    vlm_error = None

    # Tesseract on preprocessed image
    try:
        tess = TesseractOCR(lang="vie", confidence_threshold=LOW_CONFIDENCE_THRESHOLD)
        tess_result = tess.run(
            ocr_images[index],
            pdf_page_number=pdf_page,
            document_page_number=doc_page,
            page_name=page_name,
            config=psm_config,
            min_width=min_width,
            min_height=min_height,
        )
        tess_annotated = tess.annotate_image()
    except Exception as e:
        tess_error = str(e)

    # VLM on original image
    try:
        vlm = VLMOCR(model=VLM_MODEL)
        vlm_result = vlm.run(
            images[index],
            pdf_page_number=pdf_page,
            document_page_number=doc_page,
            page_name=page_name,
        )
    except Exception as e:
        vlm_error = str(e)

    # Seed editor with Tesseract baseline (or VLM fallback)
    if tess_result is not None and tess_result.paragraphs:
        baseline = "\n\n".join(p.text for p in tess_result.paragraphs)
    elif vlm_result is not None:
        baseline = "\n\n".join(p.text for p in vlm_result.paragraphs)
    else:
        baseline = ""

    return {
        "tess_result": tess_result,
        "tess_annotated": tess_annotated,
        "tess_error": tess_error,
        "vlm_result": vlm_result,
        "vlm_error": vlm_error,
        "baseline": baseline,
    }


# main


def main() -> None:
    st.header("3. OCR Review")

    psm_config, min_width, min_height, output_path = _sidebar_controls()

    # Determine which image list to use
    ocr_images: list[np.ndarray] | None = st.session_state.get("ocr_images")
    images: list[np.ndarray] = st.session_state.get("images", [])
    pages: list[int] = st.session_state.get("pages", [])

    # If no preprocessed images, fall back to original for Tesseract too
    using_preprocessed = ocr_images is not None and len(ocr_images) == len(images)
    tess_source_images = ocr_images if using_preprocessed else images

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

    # session state init
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

    # header info
    pdf_page = pages[index]
    page_adjustment = int(st.session_state.get("page_adjustment", 0))
    doc_page = pdf_page + page_adjustment
    editor_key = f"ocr_editor_widget_{index}"

    col_hdr, col_adj = st.columns([3, 1])
    with col_hdr:
        status = "✅ Confirmed" if confirmed_flags[index] else "🔲 Pending"
        st.subheader(
            f"Image {index + 1} of {total} - PDF page {pdf_page} (actual page {doc_page}) - {status}"
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

    if not using_preprocessed:
        st.warning(
            "No preprocessed images found - Tesseract will run on the original images. "
            "Apply a preprocessing pipeline on page 2 for best results."
        )

    # auto-trigger OCR if not cached
    if index not in st.session_state.ocr_page_cache:
        with st.spinner(f"Running Tesseract + VLM OCR on image {index + 1}…"):
            page_data = _run_ocr_for_index(
                index=index,
                images=images,
                ocr_images=tess_source_images,
                pages=pages,
                page_adjustment=page_adjustment,
                psm_config=psm_config,
                min_width=min_width,
                min_height=min_height,
            )
            st.session_state.ocr_page_cache[index] = page_data
            # Initialize this image's editor text with the new baseline
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

    # Ensure editor key exists for this image
    if editor_key not in st.session_state:
        st.session_state[editor_key] = page_data.get("baseline", "")

    # Ensure VLM editor key exists for this image
    if vlm_key not in st.session_state:
        st.session_state[vlm_key] = vlm_raw_baseline

    # Live computation of warnings against current editor text & current VLM reference text
    current_text = st.session_state.get(editor_key, "")
    current_vlm_text = st.session_state.get(vlm_key, "")
    warnings, mismatch_indices_by_para = _compute_warnings(
        current_vlm_text,
        current_text,
    )

    col_img, col_warn, col_text = st.columns([1.5, 1, 1])

    with col_img:
        col_mode, col_rerun = st.columns([1, 1])

        with col_mode:
            view_mode = st.radio(
                "Image display mode",
                ["Annotated", "Original image"],
                horizontal=True,
                key=f"image_view_mode_{index}",
            )

        with col_rerun:
            if st.button(
                "Re-run OCR on current image",
            ):
                st.session_state.ocr_page_cache.pop(index, None)
                st.session_state.pop(editor_key, None)
                st.session_state.pop(f"ocr_vlm_widget_{index}", None)
                st.rerun()

        if view_mode == "Annotated":
            if tess_error:
                st.error(f"Tesseract failed: {tess_error}")
            elif tess_annotated is not None:
                st.image(
                    tess_annotated,
                    width="stretch",
                )
            else:
                st.image(
                    _image_to_pil(tess_source_images[index]),
                    width="stretch",
                )
        else:
            st.image(
                _image_to_pil(images[index]),
                caption="Original image (VLM input)",
                width="stretch",
            )
        if vlm_error:
            st.error(f"VLM failed: {vlm_error}")

    with col_warn:
        if warnings:
            icons = {"mismatch": "⚡", "orphan": "🔀"}
            st.write(f"⚠️ {len(warnings)} Warning(s) & Discrepancies")
            with st.container(height=300, border=True):
                for w in warnings:
                    icon = icons.get(w["type"], "⚠️")
                    st.markdown(f"{icon} **{w['label']}** - {w['msg']}")
        else:
            st.success("✅ No mismatches detected.")

        highlighted_html = _render_highlighted_text(
            current_text, mismatch_indices_by_para
        )
        with st.container(height=200, border=True):
            st.markdown(highlighted_html, unsafe_allow_html=True)

    with col_text:
        st.badge("Separate paragraphs with a blank line.")

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

        st.subheader("Final text")

        st.text_area(
            "Reviewed text",
            height=250,
            key=editor_key,
            label_visibility="collapsed",
        )

        # navigation & confirm
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

    # finalize
    all_confirmed = all(confirmed_flags[:total])
    if all_confirmed:
        st.divider()
        st.subheader("Finalize - Write OCRDocument")
        st.success(
            f"All {total} image(s) confirmed. Ready to write the final document."
        )

        fin_col1, fin_col2 = st.columns(2)
        with fin_col1:
            source_name = st.text_input(
                "Source (PDF filename)",
                value=st.session_state.get("pdf_filename", ""),
                key="ocr_source_name",
            )
        with fin_col2:
            language = st.radio(
                "Document language",
                ["vie", "sino-nom"],
                index=0
                if st.session_state.get("document_language", "vie") == "vie"
                else 1,
                horizontal=True,
                key="ocr_final_language",
            )

        if st.button("Write OCRDocument JSON", type="primary"):
            try:
                confirmed_page_objs = [
                    OCRPage(**p)
                    for p in st.session_state.ocr_confirmed_pages
                    if p is not None
                ]
                # Determine image dimensions from first image
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
                st.success(f"OCRDocument written to `{out_path}`")
                st.download_button(
                    "Download OCRDocument JSON",
                    data=doc.model_dump_json(indent=2),
                    file_name=out_path.name,
                    mime="application/json",
                )
            except Exception as e:
                st.error(f"Failed to write OCRDocument: {e}")


if __name__ == "__main__":
    main()
