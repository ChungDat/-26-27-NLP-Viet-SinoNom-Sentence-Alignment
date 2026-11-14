from __future__ import annotations

import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import ollama
except ImportError:
    ollama = None  # type: ignore[assignment]

try:
    import pytesseract
except ImportError:
    pytesseract = None  # type: ignore[assignment]

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

TesseractData = dict[str, list[Any]]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class OCRWord(BaseModel):
    text: str
    bbox: tuple[int, int, int, int] | None = None  # (x_min, y_min, x_max, y_max)
    confidence: float | None = None


class OCRLine(BaseModel):
    line_id: int
    bbox: tuple[int, int, int, int] | None = None  # (x_min, y_min, x_max, y_max)
    text: str
    words: list[OCRWord] = Field(default_factory=list)


class OCRParagraph(BaseModel):
    paragraph_id: int
    bbox: tuple[int, int, int, int] | None = None  # (x_min, y_min, x_max, y_max)
    lines: list[OCRLine] = Field(default_factory=list)


class OCRPage(BaseModel):
    pdf_page_number: int
    document_page_number: int
    page_name: str
    method: str = "tesseract"
    paragraphs: list[OCRParagraph] = Field(default_factory=list)


class OCRDocument(BaseModel):
    source: str
    language: str
    page_count: int
    width: int | None = None
    height: int | None = None
    method: str = "tesseract"
    pages: list[OCRPage] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper: bounding-box union
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TesseractOCR
# ---------------------------------------------------------------------------


class TesseractOCR:
    """Pipeline wrapper for Tesseract OCR on a single image.

    Call :meth:`run` once per image.  After a successful :meth:`run` call the
    result is available through :attr:`page` and can be written to disk with
    :meth:`visualize`, :meth:`save_text`, and :meth:`save_json`.

    Attempting to call :meth:`visualize`, :meth:`save_text`, or
    :meth:`save_json` before :meth:`run` raises :exc:`RuntimeError`.

    Args:
        lang: Tesseract language code (e.g. ``"vie"`` for Vietnamese).
        confidence_threshold: Words with a confidence score below this value
            are drawn in red in :meth:`visualize`.  Defaults to ``85.0``.
        gap_ratio: Consecutive Tesseract paragraphs whose vertical gap is
            smaller than ``gap_ratio × median_line_height`` are merged into one
            :class:`OCRParagraph`.  Defaults to ``0.5``.
    """

    def __init__(
        self,
        lang: str,
        confidence_threshold: float = 85.0,
        gap_ratio: float = 0.5,
    ) -> None:
        self.lang = lang
        self.confidence_threshold = confidence_threshold
        self.gap_ratio = gap_ratio

        # State set by run()
        self._image_path: Path | None = None
        self._raw: TesseractData | None = None
        self._page: OCRPage | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        image_path: str | Path,
        *,
        pdf_page_number: int,
        document_page_number: int,
        page_name: str,
        config: str = "--psm 3",
    ) -> OCRPage:
        """Run the full OCR pipeline on one image and store the result.

        This method must be called before :meth:`visualize`, :meth:`save_text`,
        or :meth:`save_json`.  Calling :meth:`run` again on a different image
        replaces the previously stored result.

        Args:
            image_path: Path to the image file to process.
            pdf_page_number: 1-indexed page number as shown by the PDF reader.
            document_page_number: Actual document page number
                (PDF reader page + any page offset).
            page_name: Human-readable label for the page (e.g. ``"001.png"``).
            config: Tesseract page segmentation configuration string
                (e.g. ``"--psm 3"``).

        Returns:
            A structured :class:`OCRPage` containing all recognized paragraphs,
            lines, and words, with heuristic paragraph merging applied.
        """
        if pytesseract is None:
            raise ImportError(
                "The 'pytesseract' package is required to use TesseractOCR. "
                "Install it with: pip install pytesseract"
            )
        self._image_path = Path(image_path)
        self._raw = pytesseract.image_to_data(
            str(self._image_path),
            output_type=pytesseract.Output.DICT,
            config=config,
            lang=self.lang,
        )
        self._page = self._build_page(
            self._raw,
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

    # ------------------------------------------------------------------
    # Private pipeline helpers
    # ------------------------------------------------------------------

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
        page_name: str,
    ) -> OCRPage:
        """Convert raw Tesseract output into a structured :class:`OCRPage`.

        Words are grouped into lines and lines into paragraphs using
        Tesseract's native ``block_num`` / ``par_num`` / ``line_num`` fields.
        Adjacent paragraphs that are close enough vertically are then merged
        by :meth:`_merge_paragraphs`.

        Args:
            raw: Dictionary returned by ``pytesseract.image_to_data``.
            pdf_page_number: 1-indexed PDF reader page number.
            document_page_number: Actual document page number.
            page_name: Human-readable page label.

        Returns:
            A fully populated :class:`OCRPage`.
        """
        # Collect words grouped by (block_num, par_num, line_num)
        # Structure: {(block, par): {line_num: [OCRWord, ...]}}
        para_lines: dict[tuple[int, int], dict[int, list[OCRWord]]] = defaultdict(
            lambda: defaultdict(list)
        )

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
            line = int(raw["line_num"][i])

            word = OCRWord(
                text=text,
                bbox=(left, top, left + width, top + height),
                confidence=conf,
            )
            para_lines[(block, par)][line].append(word)

        # Build OCRParagraph list (before merging)
        paragraphs: list[OCRParagraph] = []
        para_id = 1
        for (block, par), lines_dict in sorted(para_lines.items()):
            ocr_lines: list[OCRLine] = []
            line_id = 1
            for line_num, words in sorted(lines_dict.items()):
                line_bbox = _union_bbox([w.bbox for w in words])
                line_text = " ".join(w.text for w in words)
                ocr_lines.append(
                    OCRLine(
                        line_id=line_id,
                        bbox=line_bbox,
                        text=line_text,
                        words=words,
                    )
                )
                line_id += 1

            para_bbox = _union_bbox([ln.bbox for ln in ocr_lines])
            paragraphs.append(
                OCRParagraph(
                    paragraph_id=para_id,
                    bbox=para_bbox,
                    lines=ocr_lines,
                )
            )
            para_id += 1

        merged = self._merge_paragraphs(paragraphs)

        return OCRPage(
            pdf_page_number=pdf_page_number,
            document_page_number=document_page_number,
            page_name=page_name,
            method="tesseract",
            paragraphs=merged,
        )

    def _merge_paragraphs(
        self,
        paragraphs: list[OCRParagraph],
    ) -> list[OCRParagraph]:
        """Merge consecutive paragraphs whose vertical gap is below a threshold.

        The threshold is ``gap_ratio x median_line_height``, where
        *median_line_height* is the median height of all :class:`OCRLine`
        bounding boxes across all paragraphs on this page.  Paragraphs are
        sorted top-to-bottom before merging; paragraphs whose tops overlap or
        whose gap is strictly less than the threshold are combined.

        The merged paragraph retains the ``paragraph_id`` of the first
        paragraph in the group and gets a new bounding box that is the union
        of all merged paragraphs' bounding boxes.  Line ``line_id`` values are
        renumbered sequentially within each merged paragraph.

        Args:
            paragraphs: List of :class:`OCRParagraph` objects to consider for
                merging.  May be modified in place.

        Returns:
            A new list of :class:`OCRParagraph` objects after merging.
        """
        if not paragraphs:
            return paragraphs

        # Compute median line height across all lines on this page
        all_heights = [
            ln.bbox[3] - ln.bbox[1] for para in paragraphs for ln in para.lines
        ]
        if len(all_heights) >= 2:
            median_line_height = statistics.median(all_heights)
        elif all_heights:
            median_line_height = all_heights[0]
        else:
            return paragraphs  # nothing to merge

        gap_threshold = self.gap_ratio * median_line_height

        # Sort paragraphs top-to-bottom by the top edge of their bbox
        sorted_paras = sorted(paragraphs, key=lambda p: p.bbox[1])

        merged: list[OCRParagraph] = []
        current = sorted_paras[0]

        for next_para in sorted_paras[1:]:
            current_bottom = current.bbox[3]
            next_top = next_para.bbox[1]
            gap = next_top - current_bottom

            if gap < gap_threshold:
                # Merge next_para into current
                combined_lines = current.lines + next_para.lines
                # Renumber line IDs sequentially
                for idx, ln in enumerate(combined_lines, start=1):
                    ln.line_id = idx
                current = OCRParagraph(
                    paragraph_id=current.paragraph_id,
                    bbox=_union_bbox([current.bbox, next_para.bbox]),
                    lines=combined_lines,
                )
            else:
                merged.append(current)
                current = next_para

        merged.append(current)
        return merged

    @staticmethod
    def _page_to_text(page: OCRPage) -> str:
        """Reconstruct plain text from a structured :class:`OCRPage`.

        Lines within each paragraph are joined with a newline; paragraphs
        are separated by a blank line.

        Args:
            page: The :class:`OCRPage` to convert.

        Returns:
            Plain-text representation of the page content.
        """
        para_texts = [
            "\n".join(ln.text for ln in para.lines) for para in page.paragraphs
        ]
        return "\n\n".join(para_texts)

    # ------------------------------------------------------------------
    # Output methods — require run() first
    # ------------------------------------------------------------------

    def visualize(
        self,
        destination_path: str | Path,
        save_name: str | Path,
    ) -> None:
        """Draw word-level OCR bounding boxes on the source image and save it.

        Words with confidence at or above :attr:`confidence_threshold` are
        outlined in green; words below the threshold are outlined in red.
        Each box is labelled with the recognised text and its confidence
        percentage.  The font falls back to the PIL default when the DejaVu
        Sans TrueType file is unavailable.

        Args:
            destination_path: Directory in which to write the annotated image.
                Created automatically if it does not exist.
            save_name: Filename for the annotated image (e.g. ``"viz.jpg"``).

        Raises:
            RuntimeError: If :meth:`run` has not been called yet.
        """
        self._require_run()
        assert self._image_path is not None
        assert self._raw is not None

        Path(destination_path).mkdir(parents=True, exist_ok=True)

        image = Image.open(self._image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                25,
            )
        except OSError:
            font = ImageFont.load_default()

        raw = self._raw
        for index in range(len(raw["text"])):
            recognized_text = str(raw["text"][index]).strip()
            try:
                confidence_score = float(raw["conf"][index])
            except (TypeError, ValueError):
                confidence_score = -1.0

            if not recognized_text or confidence_score < 0:
                continue

            left = int(raw["left"][index])
            top = int(raw["top"][index])
            width = int(raw["width"][index])
            height = int(raw["height"][index])

            color = (
                (0, 200, 0)
                if confidence_score >= self.confidence_threshold
                else (220, 0, 0)
            )

            draw.rectangle(
                (left, top, left + width, top + height),
                outline=color,
                width=2,
            )

            label = f"{recognized_text} ({confidence_score:.1f}%)"
            draw.text(
                (left, max(top - 30, 0)),
                label,
                font=font,
                fill=color,
            )

        image.save(str(Path(destination_path) / save_name))

    def save_text(
        self,
        destination_path: str | Path,
        save_name: str | Path,
    ) -> None:
        """Save the OCR result as a plain-text file.

        The text is reconstructed from the structured :class:`OCRPage` using
        :meth:`_page_to_text` and then passed through :func:`normalize_text`.

        Args:
            destination_path: Directory in which to write the text file.
                Created automatically if it does not exist.
            save_name: Filename for the output (e.g. ``"page_001.txt"``).

        Raises:
            RuntimeError: If :meth:`run` has not been called yet.
        """
        self._require_run()
        assert self._page is not None

        Path(destination_path).mkdir(parents=True, exist_ok=True)
        text = normalize_text(self._page_to_text(self._page))
        (Path(destination_path) / save_name).write_text(text, encoding="utf-8")

    def save_json(
        self,
        destination_path: str | Path,
        save_name: str | Path,
    ) -> None:
        """Serialize the :class:`OCRPage` result to a JSON file.

        The output format matches the schema defined by the Pydantic models
        and is compatible with the structure shown in ``template.json``.

        Args:
            destination_path: Directory in which to write the JSON file.
                Created automatically if it does not exist.
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


# ---------------------------------------------------------------------------
# Default VLM OCR Prompt
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# VLMOCR
# ---------------------------------------------------------------------------


class VLMOCR:
    """Vision-Language Model (VLM) OCR engine using Ollama.

    Transcribes an image using a multimodal model (e.g. Qwen2-VL / Qwen2.5-VL).
    Unlike traditional OCR, VLMs output raw textual transcriptions without spatial bounding boxes or per-word confidence scores.

    The extracted text is normalized with :func:`normalize_text` and structured into the unified :class:`OCRPage` schema (with ``method="vlm"``, and ``bbox=None`` for words, lines, and paragraphs), ready for downstream line/word matching with Tesseract.

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
        image_source: str | Path | bytes,
        *,
        pdf_page_number: int,
        document_page_number: int,
        page_name: str,
        options: dict[str, Any] | None = None,
    ) -> OCRPage:
        """Run VLM OCR on a single image and store the structured result.

        Args:
            image_source: Path to the image file or raw image bytes.
            pdf_page_number: 1-indexed page number from the PDF reader.
            document_page_number: Actual document page number.
            page_name: Human-readable page label (e.g. ``"0040.jpg"``).
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
            image_bytes = self._image_path.read_bytes()
        elif isinstance(image_source, bytes):
            self._image_path = None
            image_bytes = image_source
        else:
            raise TypeError(f"Unsupported image_source type: {type(image_source)}")

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

        Paragraphs are demarcated by blank lines (two or more newlines),
        lines by single newlines, and words by whitespace. Bounding boxes
        and confidences are set to None.
        """
        paragraphs: list[OCRParagraph] = []

        # Split normalized text into paragraph chunks by 2+ newlines
        raw_paragraphs = [
            p.strip() for p in re.split(r"\n\s*\n", normalized_text) if p.strip()
        ]

        for p_idx, p_text in enumerate(raw_paragraphs, start=1):
            lines: list[OCRLine] = []
            raw_lines = [ln.strip() for ln in p_text.splitlines() if ln.strip()]

            for l_idx, l_text in enumerate(raw_lines, start=1):
                words = [
                    OCRWord(text=w, bbox=None, confidence=None) for w in l_text.split()
                ]
                lines.append(
                    OCRLine(
                        line_id=l_idx,
                        bbox=None,
                        text=l_text,
                        words=words,
                    )
                )

            paragraphs.append(
                OCRParagraph(
                    paragraph_id=p_idx,
                    bbox=None,
                    lines=lines,
                )
            )

        return OCRPage(
            pdf_page_number=pdf_page_number,
            document_page_number=document_page_number,
            page_name=page_name,
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
        para_texts = [
            "\n".join(ln.text for ln in para.lines) for para in self._page.paragraphs
        ]
        text = "\n\n".join(para_texts)
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


# ---------------------------------------------------------------------------
# normalize_text — module-level function used by gui.py
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Normalize OCR text by preserving paragraph breaks and placing
    sentence-ending full stops on their own lines.

    OCR output often wraps a single sentence across multiple lines. This
    function first collapses those soft-wrap single newlines into spaces so
    that split sentences are rejoined.  Double (or more) consecutive newlines
    are retained as a single blank-line paragraph separator.

    Because the source documents may contain a space *before* a full stop
    (e.g. ``"word . Next sentence"``), the sentence-end pattern absorbs any
    optional horizontal whitespace preceding the period as well as the
    whitespace that follows it.  This avoids treating mid-line decorated dots
    as sentence boundaries while still normalising ``"word.  Next"`` and
    ``"word . Next"`` the same way.

    Finally, if the resulting text ends with a full stop (optionally preceded
    by horizontal whitespace and not already followed by a newline) a trailing
    newline is appended.

    Args:
        text: Raw OCR text, potentially containing soft-wrapped lines and
            spaces before punctuation marks.

    Returns:
        Normalized text where each sentence ending in a full stop starts on
        its own line and paragraphs are separated by a blank line.
    """
    # Step 1 — collapse single newlines (soft wraps) to spaces;
    #           collapse two-or-more newlines to a single blank line (paragraph break).
    normalized = re.sub(
        r"\n+",
        lambda m: "\n\n" if len(m.group()) >= 2 else " ",
        text,
    )

    # Step 2 — treat "[ \t]*.<whitespace>+" as a sentence boundary.
    #           The optional leading [ \t]* handles documents where a space
    #           appears before the period (e.g. "word . Next").
    #           The captured leading space (if any) is discarded so the period
    #           lands directly after the last word character.
    normalized = re.sub(r"[ \t]*\.[ \t]+", ".\n", normalized)

    # Step 3 — append a trailing newline only if the text ends with a full stop
    #           (optionally preceded by horizontal whitespace) and does not already
    #           end with a newline.  Using endswith("\n") avoids the ambiguity of
    #           re.search r"\.[ \t]*$" matching before the implicit \n that Python's
    #           $ anchor sees at the end of a string.
    if not normalized.endswith("\n") and re.search(r"[ \t]*\.$", normalized):
        normalized = re.sub(r"[ \t]*\.$", ".\n", normalized)

    return normalized


# ---------------------------------------------------------------------------
# merge_box — stub (not yet implemented)
# ---------------------------------------------------------------------------


def merge_box(data: TesseractData) -> None:
    """Merge related OCR bounding boxes using heuristic rules.

    Args:
        data: OCR data containing the bounding boxes to merge.
    """


# ---------------------------------------------------------------------------
# Module-level shims — backward compatibility with gui.py and benchmark_viet.py
# ---------------------------------------------------------------------------


def get_ocr_data(
    image_path: str | Path,
    config: str = "--psm 3",
    lang: str = "vie",
) -> TesseractData:
    """Return raw Tesseract word-level data for *image_path*.

    This is a module-level shim kept for backward compatibility with
    ``gui.py`` and ``benchmark_viet.py``.  Prefer :meth:`TesseractOCR.run`
    for new code.

    Args:
        image_path: Path to the image to process.
        config: Tesseract page segmentation configuration string.
        lang: Tesseract language code.

    Returns:
        A :class:`TesseractData` dictionary as returned by
        ``pytesseract.image_to_data``.
    """
    return pytesseract.image_to_data(
        str(image_path),
        output_type=pytesseract.Output.DICT,
        config=config,
        lang=lang,
    )


def get_ocr_text(
    image_path: str | Path,
    config: str = "--psm 3",
    lang: str = "vie",
) -> str:
    """Return the raw string recognized in *image_path* by Tesseract.

    This is a module-level shim kept for backward compatibility with
    ``benchmark_viet.py``.  Prefer :meth:`TesseractOCR.run` for new code.

    Args:
        image_path: Path to the image to process.
        config: Tesseract page segmentation configuration string.
        lang: Tesseract language code.

    Returns:
        The raw text string recognized in the image.
    """
    return pytesseract.image_to_string(
        str(image_path),
        config=config,
        lang=lang,
    )


def visualize(
    image_path: str | Path,
    data: TesseractData,
    destination_path: str | Path,
    save_name: str | Path,
    confidence_threshold: float = 85.0,
) -> None:
    """Draw word-level OCR bounding boxes on *image_path* and save the result.

    This is a module-level shim kept for backward compatibility with
    ``benchmark_viet.py``.  Prefer :meth:`TesseractOCR.visualize` for new
    code.

    Args:
        image_path: Path to the source image.
        data: OCR data returned by :func:`get_ocr_data`.
        destination_path: Directory in which to write the annotated image.
            Created automatically if it does not exist.
        save_name: Filename for the annotated image.
        confidence_threshold: Words below this confidence score are drawn in
            red; words at or above it are drawn in green.
    """
    Path(destination_path).mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            25,
        )
    except OSError:
        font = ImageFont.load_default()

    for index in range(len(data["text"])):
        recognized_text = str(data["text"][index]).strip()
        try:
            conf = float(data["conf"][index])
        except (TypeError, ValueError):
            conf = -1.0

        if not recognized_text or conf < 0:
            continue

        left = int(data["left"][index])
        top = int(data["top"][index])
        width = int(data["width"][index])
        height = int(data["height"][index])

        color = (0, 200, 0) if conf >= confidence_threshold else (220, 0, 0)

        draw.rectangle(
            (left, top, left + width, top + height),
            outline=color,
            width=2,
        )
        draw.text(
            (left, max(top - 30, 0)),
            f"{recognized_text} ({conf:.1f}%)",
            font=font,
            fill=color,
        )

    image.save(str(Path(destination_path) / save_name))


def save_text(
    text: str,
    destination_path: str | Path,
    save_name: str | Path,
) -> None:
    """Write *text* to a file inside *destination_path*.

    This is a module-level shim kept for backward compatibility with
    ``benchmark_viet.py``.  Prefer :meth:`TesseractOCR.save_text` for new
    code.

    Args:
        text: Text content to write.
        destination_path: Directory in which to write the file.
            Created automatically if it does not exist.
        save_name: Filename for the output text file.
    """
    Path(destination_path).mkdir(parents=True, exist_ok=True)
    (Path(destination_path) / save_name).write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# main — demonstration
# ---------------------------------------------------------------------------


def main() -> None:
    """Run both TesseractOCR and VLMOCR on a sample image and write outputs."""

    ocr_configs = ["--psm 3", "--psm 6"]
    source_directory = Path("Dong_hang/test")
    destination_directory = Path("Dong_hang/test")
    image_file_name = "0100.png"
    image_path = source_directory / image_file_name

    # 1. Run Tesseract OCR
    print("--- Running TesseractOCR ---")
    tesseract_ocr = TesseractOCR("vie")
    try:
        tess_page = tesseract_ocr.run(
            image_path,
            pdf_page_number=40,
            document_page_number=40,
            page_name=image_file_name,
            config=ocr_configs[0],
        )
        tesseract_ocr.visualize(destination_directory, "hypo_tesseract.jpg")
        tesseract_ocr.save_text(destination_directory, "hypo_tesseract.txt")
        tesseract_ocr.save_json(destination_directory, "hypo_tesseract.json")

        tess_line_count = sum(len(p.lines) for p in tess_page.paragraphs)
        print(
            f"[Tesseract] Extracted {tess_line_count} line(s) "
            f"across {len(tess_page.paragraphs)} paragraph(s)."
        )
    except Exception as exc:
        print(f"[Tesseract] Failed: {exc}")

    # 2. Run VLM OCR
    print("\n--- Running VLMOCR ---")
    vlm_ocr = VLMOCR()
    try:
        vlm_page = vlm_ocr.run(
            image_path,
            pdf_page_number=40,
            document_page_number=40,
            page_name=image_file_name,
        )
        vlm_ocr.save_text(destination_directory, "hypo_vlm.txt")
        vlm_ocr.save_json(destination_directory, "hypo_vlm.json")

        vlm_line_count = sum(len(p.lines) for p in vlm_page.paragraphs)
        print(
            f"[VLM] Extracted {vlm_line_count} line(s) "
            f"across {len(vlm_page.paragraphs)} paragraph(s)."
        )
        if vlm_ocr.raw_response and hasattr(vlm_ocr.raw_response, "eval_count"):
            print(
                f"[VLM] Prompt tokens: {vlm_ocr.raw_response.prompt_eval_count}, "
                f"Generated tokens: {vlm_ocr.raw_response.eval_count}"
            )
    except Exception as exc:
        print(f"[VLM] Failed: {exc}")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"Execution time: {(end - start) * 1000:.2f} s")
