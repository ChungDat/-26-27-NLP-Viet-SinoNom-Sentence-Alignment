from __future__ import annotations

import io
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import ollama
import pytesseract
from PIL import Image, ImageDraw, ImageFont

try:
    from .schemas import (
        OCRPage,
        OCRParagraph,
    )
except ImportError:
    from schemas import (  # type: ignore[no-redef]
        OCRPage,
        OCRParagraph,
    )

TesseractData = dict[str, list[Any]]


def to_pil_image(image: str | Path | np.ndarray | Image.Image | bytes) -> Image.Image:
    """Convert any supported image source to an RGB PIL Image."""
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, bytes):
        return Image.open(io.BytesIO(image)).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if np is not None and isinstance(image, np.ndarray):
        if cv2 is not None and image.ndim == 3 and image.shape[2] == 4:
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGBA2RGB))
        return Image.fromarray(image).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def to_image_bytes(image: str | Path | bytes | np.ndarray | Image.Image) -> bytes:
    """Convert any supported image source to PNG encoded bytes."""
    if isinstance(image, (str, Path)):
        return Path(image).read_bytes()
    if isinstance(image, bytes):
        return image
    if np is not None and isinstance(image, np.ndarray):
        if cv2 is not None:
            if image.ndim == 2:
                bgr = image
            elif image.shape[2] == 4:
                bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            else:
                bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            success, encoded = cv2.imencode(".png", bgr)
            if not success:
                raise ValueError("Could not encode NumPy array to PNG bytes")
            return encoded.tobytes()
        buf = io.BytesIO()
        to_pil_image(image).save(buf, format="PNG")
        return buf.getvalue()
    if isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
    raise TypeError(f"Unsupported image type: {type(image)}")


def normalize_text(text: str) -> str:
    """Normalize OCR text by collapsing soft line wraps and preserving paragraph breaks

    Single newlines are treated as OCR soft wraps and replaced with spaces; two or more consecutive newlines remain a single blank-line paragraph separator. Sentence punctuation does not create line breaks.

    Args:
        text: Raw OCR text, potentially containing soft-wrapped lines and spaces before punctuation marks.

    Returns:
        Normalized text with soft wraps collapsed and paragraphs separated by a blank line.
    """
    # Step 1 — collapse single newlines (soft wraps) to spaces;
    #           collapse two-or-more newlines to a single blank line (paragraph break).
    normalized = re.sub(
        r"\n+",
        lambda m: "\n\n" if len(m.group()) >= 2 else " ",
        text,
    )

    normalized = re.sub(r"[ \t]+\.", ".", normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = normalized.strip(" \t")

    # Append a trailing newline only if the text ends with a full stop.
    if not normalized.endswith("\n") and re.search(r"[ \t]*\.$", normalized):
        normalized = re.sub(r"[ \t]*\.$", ".\n", normalized)

    return normalized


def filtered_ocr_text(data: TesseractData) -> str:
    """Rebuild text from retained OCR word boxes in Tesseract reading order.

    Groups words into paragraphs according to block_num and par_num, ignoring line boundaries, then normalizes the resulting text.

    Args:
        data: TesseractData dictionary.

    Returns:
        Normalized text string.
    """
    paragraphs: list[str] = []
    words: list[str] = []
    current_paragraph: tuple[int, int] | None = None

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

    def finish_paragraph() -> None:
        nonlocal words
        if not words:
            return
        paragraphs.append(" ".join(words))
        words = []

    for index, raw_text in enumerate(data.get("text", [])):
        text = str(raw_text).strip()
        if not text or confidence(index) < 0:
            continue

        paragraph = (value("block_num", index), value("par_num", index))
        if current_paragraph is not None and paragraph != current_paragraph:
            finish_paragraph()
        current_paragraph = paragraph
        words.append(text)

    finish_paragraph()
    return normalize_text("\n\n".join(paragraphs))


def _union_bbox(
    boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    """Return the axis-aligned bounding box that contains all given boxes.

    Args:
        boxes: Non-empty list of ``(x_min, y_min, x_max, y_max)`` tuples.

    Returns:
        A single ``(x_min, y_min, x_max, y_max)`` tuple covering all inputs.

    Raises:
        ValueError: If *boxes* is empty.
    """
    if not boxes:
        raise ValueError("boxes must not be empty")
    x_mins, y_mins, x_maxs, y_maxs = zip(*boxes)
    return (min(x_mins), min(y_mins), max(x_maxs), max(y_maxs))


class TesseractOCR:
    """Pipeline wrapper for Tesseract OCR on a single image.

    Call :meth:`run` once per image.  After a successful :meth:`run` call the result is available through :attr:`page` and can be written to disk with :meth:`visualize`, :meth:`save_text`, and :meth:`save_json`.

    Attempting to call :meth:`visualize`, :meth:`save_text`, or :meth:`save_json` before :meth:`run` raises :exc:`RuntimeError`.

    Args:
        lang: Tesseract language code (e.g. ``"vie"`` for Vietnamese).
        confidence_threshold: Words with a confidence score below this value are drawn in red in :meth:`visualize`.  Defaults to ``85.0``.
        gap_ratio: Blocks and paragraphs whose vertical gap is smaller than ``gap_ratio * median_word_height`` are merged. Defaults to ``0.5``.
    """

    def __init__(
        self,
        lang: str = "vie",
        confidence_threshold: float = 85.0,
        gap_ratio: float = 0.5,
    ) -> None:
        self.lang = lang
        self.confidence_threshold = confidence_threshold
        self.gap_ratio = gap_ratio

        # State set by run()
        self._image_path: Path | None = None
        self._image: Image.Image | None = None
        self._raw: TesseractData | None = None
        self._filtered_raw: TesseractData | None = None
        self._page: OCRPage | None = None

    def run(
        self,
        image: str | Path | np.ndarray | Image.Image | bytes,
        *,
        pdf_page_number: int = 1,
        document_page_number: int = 1,
        page_name: str = "image",
        config: str = "--psm 3",
        min_width: int = 0,
        min_height: int = 0,
    ) -> OCRPage:
        """Run the full OCR pipeline on one image and store the result.

        This method must be called before :meth:`visualize`, :meth:`save_text`, or :meth:`save_json`.  Calling :meth:`run` again on a different image replaces the previously stored result.

        Args:
            image: Source image as a file path, NumPy array, PIL Image, or bytes.
            pdf_page_number: 1-indexed page number as shown by the PDF reader.
            document_page_number: Actual document page number (PDF reader page + any page offset).
            page_name: Human-readable label for the page (e.g. ``"001.png"``).
            config: Tesseract page segmentation configuration string (e.g. ``"--psm 3"``).
            min_width: Minimum bounding-box width in pixels (boxes smaller are excluded).
            min_height: Minimum bounding-box height in pixels (boxes smaller are excluded).

        Returns:
            A structured :class:`OCRPage` containing recognized paragraphs and words, with heuristic paragraph merging applied.
        """

        if isinstance(image, (str, Path)):
            self._image_path = Path(image)
        else:
            self._image_path = None

        self._image = to_pil_image(image)
        self._raw = pytesseract.image_to_data(
            self._image,
            output_type=pytesseract.Output.DICT,
            config=config,
            lang=self.lang,
        )

        if min_width > 0 or min_height > 0:
            self._filtered_raw = self.filter_bounding_boxes(
                self._raw,
                min_width=min_width,
                min_height=min_height,
            )
        else:
            self._filtered_raw = self._raw

        self._page = self._build_page(
            self._filtered_raw,
            pdf_page_number=pdf_page_number,
            document_page_number=document_page_number,
            page_name=page_name,
        )
        return self._page

    @property
    def page(self) -> OCRPage:
        """The :class:`OCRPage` produced by the most recent :meth:`run` call.

        Raises:
            RuntimeError: If :meth:`run` has not been called yet.
        """
        self._require_run()
        return self._page  # type: ignore[return-value]

    @property
    def text(self) -> str:
        """The normalized plain-text produced by the most recent :meth:`run` call.

        Raises:
            RuntimeError: If :meth:`run` has not been called yet.
        """
        self._require_run()
        return normalize_text(self._page_to_text(self._page))  # type: ignore[arg-type]

    @property
    def raw(self) -> TesseractData:
        """The raw Tesseract dictionary produced by the most recent :meth:`run` call."""
        self._require_run()
        return self._raw  # type: ignore[return-value]

    @property
    def filtered_raw(self) -> TesseractData:
        """The filtered Tesseract dictionary after bounding-box filtering."""
        self._require_run()
        return self._filtered_raw if self._filtered_raw is not None else self._raw  # type: ignore[return-value]

    def _require_run(self) -> None:
        """Raise :exc:`RuntimeError` when :meth:`run` has not been called."""
        if self._page is None:
            raise RuntimeError(
                "No OCR result available. Call run() before visualize(), "
                "save_text(), or save_json()."
            )

    def _build_page(
        self,
        raw: TesseractData,
        *,
        pdf_page_number: int,
        document_page_number: int,
        page_name: str | Path,
    ) -> OCRPage:
        """Convert raw Tesseract output into a structured :class:`OCRPage`.

        Words are grouped by Tesseract's native block and paragraph numbers. Raw line boundaries are discarded. Nearby blocks and then nearby paragraphs are merged by :meth:`_merge_paragraphs`.

        Args:
            raw: Dictionary returned by ``pytesseract.image_to_data``.
            pdf_page_number: 1-indexed PDF reader page number.
            document_page_number: Actual document page number.
            page_name: Human-readable page label.

        Returns:
            A fully populated :class:`OCRPage`.
        """
        # Tesseract's word sequence is already in reading order. Its line
        # numbers are intentionally ignored because line wraps are not model
        # boundaries.
        # Store words as plain (text, bbox) tuples — not persisted long-term.
        block_paragraph_words: dict[
            int, dict[int, list[tuple[str, tuple[int, int, int, int]]]]
        ] = defaultdict(lambda: defaultdict(list))

        n = len(raw.get("text", []))
        for i in range(n):
            text = str(raw["text"][i]).strip()
            conf_raw = raw["conf"][i]
            try:
                conf = float(conf_raw)
            except (TypeError, ValueError):
                conf = -1.0

            if not text or conf < 0:
                continue

            left = int(raw["left"][i])
            top = int(raw["top"][i])
            width = int(raw["width"][i])
            height = int(raw["height"][i])
            block = int(raw["block_num"][i])
            par = int(raw["par_num"][i])
            block_paragraph_words[block][par].append(
                (text, (left, top, left + width, top + height))
            )

        # Build paragraph groups within native Tesseract blocks.
        blocks: list[list[OCRParagraph]] = []
        para_id = 1
        all_word_heights: list[int] = []
        for block_num in sorted(block_paragraph_words):
            paragraphs: list[OCRParagraph] = []
            for par_num in sorted(block_paragraph_words[block_num]):
                word_tuples = block_paragraph_words[block_num][par_num]
                all_word_heights.extend(bbox[3] - bbox[1] for _, bbox in word_tuples)
                para_bbox = _union_bbox([bbox for _, bbox in word_tuples])
                paragraphs.append(
                    OCRParagraph(
                        paragraph_id=para_id,
                        bbox=para_bbox,
                        text=" ".join(text for text, _ in word_tuples),
                    )
                )
                para_id += 1
            blocks.append(paragraphs)

        if len(all_word_heights) >= 2:
            median_word_height = float(statistics.median(all_word_heights))
        elif all_word_heights:
            median_word_height = float(all_word_heights[0])
        else:
            median_word_height = None

        merged = self._merge_paragraphs(blocks, median_word_height=median_word_height)

        return OCRPage(
            pdf_page_number=pdf_page_number,
            document_page_number=document_page_number,
            page_name=str(page_name),
            method="tesseract",
            paragraphs=merged,
        )

    def _merge_paragraphs(
        self,
        blocks: list[list[OCRParagraph]],
        median_word_height: float | None = None,
    ) -> list[OCRParagraph]:
        """Merge nearby blocks, then nearby paragraphs within each block.

        Both passes use ``gap_ratio * median_word_height``, where *median_word_height* is the median height of all word bounding boxes on the page. A block's bounds are the union of its paragraph bounds. Blocks and paragraphs are sorted top-to-bottom, and adjacent regions whose vertical gap is strictly less than the threshold are combined.

        A merged paragraph retains the first paragraph's ID and combines its ordered words with the words of the following paragraph.

        Args:
            blocks: Paragraph groups organized by their original Tesseract block number.
            median_word_height: Median height of all word bounding boxes on the page. If omitted or None, falls back to paragraph heights.

        Returns:
            A flat list of paragraphs after both merge passes.
        """
        non_empty_blocks = [block for block in blocks if block]
        if not non_empty_blocks:
            return []

        if median_word_height is None:
            all_heights = [
                bbox[3] - bbox[1]
                for block in non_empty_blocks
                for para in block
                if para.bbox is not None
                for bbox in [para.bbox]
            ]
            if len(all_heights) >= 2:
                median_word_height = float(statistics.median(all_heights))
            elif all_heights:
                median_word_height = float(all_heights[0])
            else:
                return [para for block in non_empty_blocks for para in block]

        gap_threshold = self.gap_ratio * median_word_height

        def group_bbox(paragraphs: list[OCRParagraph]) -> tuple[int, int, int, int]:
            return _union_bbox(
                [para.bbox for para in paragraphs if para.bbox is not None]
            )

        # Merge nearby Tesseract blocks first.
        sorted_blocks = sorted(
            non_empty_blocks,
            key=lambda block: (
                group_bbox(block)[1],
                group_bbox(block)[0],
            ),
        )
        merged_blocks: list[list[OCRParagraph]] = []
        current_block = list(sorted_blocks[0])
        current_block_bbox = group_bbox(current_block)

        for next_block in sorted_blocks[1:]:
            next_block_bbox = group_bbox(next_block)
            gap = next_block_bbox[1] - current_block_bbox[3]
            if gap < gap_threshold:
                current_block.extend(next_block)
                current_block_bbox = _union_bbox([current_block_bbox, next_block_bbox])
            else:
                merged_blocks.append(current_block)
                current_block = list(next_block)
                current_block_bbox = next_block_bbox
        merged_blocks.append(current_block)

        # Then merge nearby paragraphs within each resulting block.
        merged_paragraphs: list[OCRParagraph] = []
        for block in merged_blocks:
            sorted_paragraphs = sorted(
                block,
                key=lambda para: (
                    para.bbox[1] if para.bbox is not None else 0,
                    para.bbox[0] if para.bbox is not None else 0,
                ),
            )
            current = sorted_paragraphs[0]

            for next_para in sorted_paragraphs[1:]:
                current_bbox = current.bbox
                next_bbox = next_para.bbox
                gap = (
                    next_bbox[1] - current_bbox[3]
                    if current_bbox is not None and next_bbox is not None
                    else float("inf")
                )

                if gap < gap_threshold:
                    combined_bbox = (
                        _union_bbox([current_bbox, next_bbox])
                        if current_bbox is not None and next_bbox is not None
                        else current_bbox or next_bbox
                    )
                    current = OCRParagraph(
                        paragraph_id=current.paragraph_id,
                        bbox=combined_bbox,
                        text=" ".join([current.text, next_para.text]),
                    )
                else:
                    merged_paragraphs.append(current)
                    current = next_para

            merged_paragraphs.append(current)
        return merged_paragraphs

    def filter_bounding_boxes(
        self,
        data: TesseractData,
        min_width: int = 0,
        min_height: int = 0,
    ) -> TesseractData:
        """Return OCR data containing only boxes that meet both size limits.

        A box is rejected when either its width or height is below the requested minimum. Every list-valued OCR field is filtered by the same indices, so its values remain aligned with the retained ``text`` entries.

        Args:
            data: Raw TesseractData dictionary returned by pytesseract.
            min_width: Minimum box width in pixels.
            min_height: Minimum box height in pixels.

        Returns:
            Filtered TesseractData dictionary.

        Raises:
            ValueError: If min_width or min_height is negative.
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

        filtered_data: TesseractData = {}
        for field, values in data.items():
            if isinstance(values, list):
                filtered_data[field] = [
                    values[index] for index in retained_indices if index < len(values)
                ]
            else:
                filtered_data[field] = values
        return filtered_data

    @staticmethod
    def _page_to_text(page: OCRPage) -> str:
        """Reconstruct plain text from a structured :class:`OCRPage`.

        Paragraphs are separated by a blank line; words within each paragraph
        are joined with spaces.

        Args:
            page: The :class:`OCRPage` to convert.

        Returns:
            Plain-text representation of the page content.
        """
        return "\n\n".join(para.text for para in page.paragraphs)

    def annotate_image(
        self,
        # font_path: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        font_size: int = 25,
        box_width: int = 2,
        high_conf_color: tuple[int, int, int] = (0, 191, 255),
        low_conf_color: tuple[int, int, int] = (255, 0, 0),
    ) -> Image.Image:
        """Draw word-level OCR bounding boxes on an image, marking low-confidence detections.

        Args:
            confidence_threshold: Minimum confidence score for high-confidence highlighting.
            font_path: Path to TrueType font file. Falls back to default font if unavailable.
            font_size: Font size for bounding box labels.
            box_width: Thickness of bounding box borders.
            high_conf_color: RGB tuple for boxes with confidence >= threshold.
            low_conf_color: RGB tuple for boxes with confidence < threshold.

        Returns:
            Annotated PIL Image.

        Raises:
            RuntimeError: If :meth:`run` has not been called yet.
        """
        self._require_run()
        assert self._image is not None

        pil_image = to_pil_image(self._image).copy()
        draw = ImageDraw.Draw(pil_image)

        try:
            font_path = Path(__file__).parent.parent / "fonts" / "NotoSans-Regular.ttf"
            font = ImageFont.truetype(font_path, font_size)
        except OSError:
            font = ImageFont.load_default(font_size)

        data = self._filtered_raw or self._raw
        text_entries = data.get("text", [])
        conf_entries = data.get("conf", [])
        left_entries = data.get("left", [])
        top_entries = data.get("top", [])
        width_entries = data.get("width", [])
        height_entries = data.get("height", [])
        # paragraph_entries = data.get("par_num", [])
        # block_entries = data.get("block_num", [])
        # level_entries = data.get("level", [])

        for index in range(len(text_entries)):
            recognized_text = str(text_entries[index]).strip()
            try:
                conf = float(conf_entries[index])
            except (IndexError, TypeError, ValueError):
                conf = -1.0

            if not recognized_text or conf <= 0:
                continue

            try:
                left = int(left_entries[index])
                top = int(top_entries[index])
                width = int(width_entries[index])
                height = int(height_entries[index])
            except (IndexError, TypeError, ValueError):
                continue

            color = (
                high_conf_color if conf >= self.confidence_threshold else low_conf_color
            )

            draw.rectangle(
                (left, top, left + width, top + height),
                outline=color,
                width=box_width,
            )

            # label = f"{recognized_text} ({conf:.1f}%)"
            label = ""
            # label = f"{str(block_entries[index]).strip()}, {str(paragraph_entries[index]).strip()}"
            # label = ""
            draw.text(
                (left, max(top - font_size - 5, 0)),
                label,
                font=font,
                fill=color,
            )

        return pil_image

    def annotate_paragraphs(
        self,
        box_width: int = 2,
        color: tuple[int, int, int] = (0, 255, 0),
        font_size: int = 20,
    ) -> Image.Image:
        """Draw paragraph-level bounding boxes on an image.

        Args:
            box_width: Thickness of bounding box borders.
            color: RGB tuple for paragraph bounding boxes. Defaults to green.
            font_size: Font size for paragraph ID labels.

        Returns:
            Annotated PIL Image.

        Raises:
            RuntimeError: If :meth:`run` has not been called yet.
        """
        self._require_run()
        assert self._image is not None
        assert self._page is not None

        pil_image = to_pil_image(self._image).copy()
        draw = ImageDraw.Draw(pil_image)

        try:
            font_path = Path(__file__).parent.parent / "fonts" / "NotoSans-Regular.ttf"
            font = ImageFont.truetype(font_path, font_size)
        except OSError:
            font = ImageFont.load_default(font_size)

        for p in self._page.paragraphs:
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

    def visualize(
        self,
        destination_path: str | Path,
        save_name: str | Path,
    ) -> None:
        """Draw word-level OCR bounding boxes on the source image and save it.

        Args:
            destination_path: Directory in which to write the annotated image. Created automatically if it does not exist.
            save_name: Filename for the annotated image.
            confidence_threshold: Optional threshold override.

        Raises:
            RuntimeError: If :meth:`run` has not been called yet.
        """
        self._require_run()
        Path(destination_path).mkdir(parents=True, exist_ok=True)
        annotated = self.annotate_image()
        annotated.save(str(Path(destination_path) / save_name))

    def save_text(
        self,
        destination_path: str | Path,
        save_name: str | Path,
    ) -> None:
        """Save the OCR result as a plain-text file.

        The text is reconstructed from the structured :class:`OCRPage` using :meth:`_page_to_text` and then passed through :func:`normalize_text`.

        Args:
            destination_path: Directory in which to write the text file. Created automatically if it does not exist.
            save_name: Filename for the output (e.g. ``"page_001.txt"``).

        Raises:
            RuntimeError: If :meth:`run` has not been called yet.
        """
        self._require_run()
        assert self._page is not None

        Path(destination_path).mkdir(parents=True, exist_ok=True)
        (Path(destination_path) / save_name).write_text(self.text, encoding="utf-8")

    def save_json(
        self,
        destination_path: str | Path,
        save_name: str | Path,
    ) -> None:
        """Serialize the :class:`OCRPage` result to a JSON file.

        The output format matches the schema defined by the Pydantic models and is compatible with the structure shown in ``template.json``.

        Args:
            destination_path: Directory in which to write the JSON file. Created automatically if it does not exist.
            save_name: Filename for the JSON output (e.g. ``"page_001.json"``).

        Raises:
            RuntimeError: If :meth:`run` has not been called yet.
        """
        self._require_run()
        assert self._page is not None

        Path(destination_path).mkdir(parents=True, exist_ok=True)
        (Path(destination_path) / save_name).write_text(
            self._page.model_dump_json(indent=2),
            encoding="utf-8",
        )


DEFAULT_VLM_PROMPT = """You are an OCR (Optical Character Recognition) engine. Your task is to extract all visible text from the uploaded image as accurately and completely as possible.

1. Transcribe all visible Vietnamese text exactly as it appears.
2. Preserve the original language, wording, spelling, punctuation, and diacritics.
3. Do not translate, paraphrase, summarize, or interpret the content.
4. Do not automatically correct spelling or grammar.
5. Never repeat a sentence, paragraph, line, or page section.

6. Extract all readable text, including titles, headings, paragraphs, tables, labels, footnotes, page numbers, and handwritten text when legible.
7. Do not skip text because it appears insignificant or repetitive.
8. Preserve the reading order according to the document layout.
9. For multi-column documents, process columns in their natural reading order.

10. Preserve paragraph boundaries and line breaks where meaningful.
11. Preserve lists and numbering.
12. Represent tables using Markdown tables when their structure is clear.
13. Preserve punctuation and spacing as accurately as possible.

14. Never invent missing or unreadable characters.
15. If a character or word cannot be confidently recognized, mark it as [UNCLEAR].
16. If only part of a word is unreadable, preserve the readable portion and mark the uncertain portion.
17. Do not guess text based on context.

18. Ignore visual decorations, logos, and non-textual elements unless they contain readable text.
19. Do not describe the image or its visual appearance.

20. Return only the extracted text.
21. Do not include explanations, introductions, conclusions, confidence assessments, or additional commentary.

IMPORTANT:
Once every visible text region has been transcribed,
STOP GENERATING IMMEDIATELY.
"""


class VLMOCR:
    """Vision-Language Model (VLM) OCR engine using Ollama.

    Transcribes an image using a multimodal model (e.g. Qwen2-VL / Qwen2.5-VL).

    The extracted text is normalized with :func:`normalize_text` and structured into the unified :class:`OCRPage` schema (with ``method="vlm"`` and ``bbox=None`` for words and paragraphs).

    Args:
        model: Name of the Ollama vision-language model. Defaults to ``"qwen3-vl:2b-instruct"``.
        prompt: System / instruction prompt guiding transcription. Defaults to :const:`DEFAULT_VLM_PROMPT`.
        options: Default generation parameters passed to Ollama (temperature, num_predict, etc.).
        host: Optional Ollama host URL (e.g. ``"http://localhost:11434"``).
    """

    DEFAULT_OPTIONS: dict[str, Any] = {
        "temperature": 0.1,
        "top_p": 0.9,
        "num_ctx": 8192,
        "repeat_penalty": 1.15,
        "num_predict": 2048,
    }

    def __init__(
        self,
        model: str = "qwen3-vl:2b-instruct",
        prompt: str = DEFAULT_VLM_PROMPT,
        options: dict[str, Any] | None = None,
        host: str | None = None,
    ) -> None:
        self.model = model
        self.prompt = prompt
        self.options = {**self.DEFAULT_OPTIONS, **(options or {})}
        self.host = host

        # State populated by run()
        self._image_path: Path | None = None
        self._raw_response: Any = None
        self._raw_text: str | None = None
        self._page: OCRPage | None = None

    def run(
        self,
        image_source: str | Path | bytes | np.ndarray | Image.Image,
        *,
        pdf_page_number: int = 1,
        document_page_number: int = 1,
        page_name: str = "image",
        options: dict[str, Any] | None = None,
    ) -> OCRPage:
        """Run VLM OCR on a single image and store the structured result.

        Args:
            image_source: Path to the image file, raw image bytes, NumPy array, or PIL Image.
            pdf_page_number: 1-indexed page number from the PDF reader.
            document_page_number: Actual document page number.
            page_name: Human-readable page label.
            options: Runtime Ollama option overrides for this specific call.

        Returns:
            A structured :class:`OCRPage` with ``method="vlm"``.
        """
        if ollama is None:
            raise ImportError(
                "The 'ollama' package is required to use VLMOCR. "
                "Install it with: pip install ollama"
            )

        if isinstance(image_source, (str, Path)):
            self._image_path = Path(image_source)
        else:
            self._image_path = None

        image_bytes = to_image_bytes(image_source)

        call_options = {**self.options, **(options or {})}

        client = ollama.Client(host=self.host) if self.host else ollama
        response = client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": self.prompt,
                    "images": [image_bytes],
                }
            ],
            options=call_options,
        )

        self._raw_response = response
        self._raw_text = response.message.content
        normalized = normalize_text(self._raw_text)

        self._page = self._build_page(
            normalized_text=normalized,
            pdf_page_number=pdf_page_number,
            document_page_number=document_page_number,
            page_name=page_name,
        )
        return self._page

    @property
    def page(self) -> OCRPage:
        """The :class:`OCRPage` produced by the most recent :meth:`run` call.

        Raises:
            RuntimeError: If :meth:`run` has not been called yet.
        """
        self._require_run()
        return self._page  # type: ignore[return-value]

    @property
    def text(self) -> str:
        """The normalized plain-text produced by the VLM.

        Raises:
            RuntimeError: If :meth:`run` has not been called yet.
        """
        self._require_run()
        assert self._page is not None
        return normalize_text("\n\n".join(para.text for para in self._page.paragraphs))

    @property
    def raw_text(self) -> str:
        """The raw un-normalized text string returned by the VLM.

        Raises:
            RuntimeError: If :meth:`run` has not been called yet.
        """
        self._require_run()
        return self._raw_text  # type: ignore[return-value]

    @property
    def raw_response(self) -> Any:
        """The full response object returned by Ollama (including token counts and timing).

        Raises:
            RuntimeError: If :meth:`run` has not been called yet.
        """
        self._require_run()
        return self._raw_response

    def _require_run(self) -> None:
        """Raise RuntimeError when run() has not been called."""
        if self._page is None:
            raise RuntimeError(
                "No VLM OCR result available. Call run() before save_text() or save_json()."
            )

    def _build_page(
        self,
        normalized_text: str,
        pdf_page_number: int,
        document_page_number: int,
        page_name: str,
    ) -> OCRPage:
        """Parse normalized VLM text into a structured :class:`OCRPage`.

        Paragraphs are demarcated by blank lines (two or more newlines). All whitespace within a paragraph separates words; bounding boxes and confidences are set to None.
        """
        paragraphs: list[OCRParagraph] = []

        # Split normalized text into paragraph chunks by 2+ newlines
        raw_paragraphs = [
            p.strip() for p in re.split(r"\n\s*\n", normalized_text) if p.strip()
        ]

        for p_idx, p_text in enumerate(raw_paragraphs, start=1):
            paragraphs.append(
                OCRParagraph(
                    paragraph_id=p_idx,
                    bbox=None,
                    text=" ".join(p_text.split()),
                )
            )

        return OCRPage(
            pdf_page_number=pdf_page_number,
            document_page_number=document_page_number,
            page_name=str(page_name),
            method="vlm",
            paragraphs=paragraphs,
        )

    def save_text(
        self,
        destination_path: str | Path,
        save_name: str | Path,
    ) -> None:
        """Save normalized VLM text to a file.

        Args:
            destination_path: Directory in which to write the text file.
            save_name: Filename for the output (e.g. ``"page_001_vlm.txt"``).

        Raises:
            RuntimeError: If :meth:`run` has not been called yet.
        """
        self._require_run()
        assert self._page is not None

        Path(destination_path).mkdir(parents=True, exist_ok=True)
        text = self.text
        if text and not text.endswith("\n"):
            text += "\n"
        (Path(destination_path) / save_name).write_text(text, encoding="utf-8")

    def save_json(
        self,
        destination_path: str | Path,
        save_name: str | Path,
    ) -> None:
        """Serialize the :class:`OCRPage` VLM result to a JSON file.

        Args:
            destination_path: Directory in which to write the JSON file.
            save_name: Filename for the output (e.g. ``"page_001_vlm.json"``).

        Raises:
            RuntimeError: If :meth:`run` has not been called yet.
        """
        self._require_run()
        assert self._page is not None

        Path(destination_path).mkdir(parents=True, exist_ok=True)
        (Path(destination_path) / save_name).write_text(
            self._page.model_dump_json(indent=2),
            encoding="utf-8",
        )


def main() -> None:
    """Run both TesseractOCR and VLMOCR on a sample image and write outputs."""

    ocr_configs = ["--psm 3", "--psm 6"]
    source_directory = Path("Dong_hang/test")
    destination_directory = Path("Dong_hang/test")
    image_file_name = Path("T1_Sách Truyện Các Thánh (Tháng 1).021.jpg")
    image_path = source_directory / image_file_name

    # 1. Run Tesseract OCR
    print("--- Running TesseractOCR ---")
    tesseract_ocr = TesseractOCR("vie")
    try:
        tess_page = tesseract_ocr.run(
            image_path,
            pdf_page_number=21,
            document_page_number=21,
            page_name=image_file_name,
            config=ocr_configs[0],
        )
        tesseract_ocr.visualize(
            destination_directory, image_file_name.stem + "_tesseract.png"
        )
        tesseract_ocr.save_text(
            destination_directory, image_file_name.stem + "_tesseract.txt"
        )
        tesseract_ocr.save_json(
            destination_directory, image_file_name.stem + "_tesseract.json"
        )

        tess_word_count = sum(len(p.text.split()) for p in tess_page.paragraphs)
        print(
            f"[Tesseract] Extracted {tess_word_count} word(s) "
            f"across {len(tess_page.paragraphs)} paragraph(s)."
        )
    except Exception as exc:
        print(f"[Tesseract] Failed: {exc}")

    # # 2. Run VLM OCR
    # print("\n--- Running VLMOCR ---")
    # vlm_ocr = VLMOCR()
    # try:
    #     vlm_page = vlm_ocr.run(
    #         image_path,
    #         pdf_page_number=15,
    #         document_page_number=15,
    #         page_name=image_file_name,
    #     )
    #     vlm_ocr.save_text(destination_directory, image_file_name.stem + "_vlm.txt")
    #     vlm_ocr.save_json(destination_directory, image_file_name.stem + "_vlm.json")

    #     vlm_word_count = sum(len(p.text.split()) for p in vlm_page.paragraphs)
    #     print(
    #         f"[VLM] Extracted {vlm_word_count} word(s) "
    #         f"across {len(vlm_page.paragraphs)} paragraph(s)."
    #     )
    #     if vlm_ocr.raw_response and hasattr(vlm_ocr.raw_response, "eval_count"):
    #         print(
    #             f"[VLM] Prompt tokens: {vlm_ocr.raw_response.prompt_eval_count}, "
    #             f"Generated tokens: {vlm_ocr.raw_response.eval_count}"
    #         )
    # except Exception as exc:
    #     print(f"[VLM] Failed: {exc}")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"Execution time: {(end - start) * 1000:.2f} s")
