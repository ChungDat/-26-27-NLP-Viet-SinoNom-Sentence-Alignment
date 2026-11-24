"""PPOCRLabel serialization, parsing, and conversion utilities.

This module provides:
- Pydantic models for the PPOCRLabel annotation format.
- Helpers to build image paths from document metadata.
- A converter from OCRPage to PPOCRLabelPage.
- Writers for Label.txt and fileState.txt.
- A parser for existing Label.txt files.
- A one-time migration helper for legacy OCRDocument JSON files.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator, model_validator

try:
    from .schemas import OCRPage
except ImportError:
    from schemas import OCRPage  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PPOCRLabelAnnotation(BaseModel):
    """A single PPOCRLabel annotation with transcription and four bounding-box points.

    Points must be in top-left → top-right → bottom-right → bottom-left order.
    Each point is a two-element ``[x, y]`` list of integers.

    Attributes:
        transcription: The text content of the annotation (NFC-normalized, non-empty).
        points: Exactly four ``[x, y]`` integer coordinate pairs.
        align_to: Optional document page number for later alignment stage.
    """

    transcription: str
    points: list[list[int]]
    document_page_number: int | None = None
    align_to: int | None = None

    @field_validator("transcription")
    @classmethod
    def _validate_transcription(cls, v: str) -> str:
        normalized = unicodedata.normalize("NFC", v).replace("\n", " ").strip()
        if not normalized:
            raise ValueError("transcription must be non-empty after NFC normalization")
        return normalized

    @field_validator("points")
    @classmethod
    def _validate_points(cls, v: list[list[int]]) -> list[list[int]]:
        if len(v) != 4:
            raise ValueError(f"points must contain exactly 4 pairs, got {len(v)}")
        for i, point in enumerate(v):
            if len(point) != 2:
                raise ValueError(
                    f"points[{i}] must be a 2-element [x, y] list, got {point!r}"
                )
            if not all(isinstance(c, int) for c in point):
                raise ValueError(
                    f"points[{i}] coordinates must be integers, got {point!r}"
                )
        return v


class PPOCRLabelPage(BaseModel):
    """A PPOCRLabel page entry mapping an image path to its annotations.

    Attributes:
        image_path: Relative image path in the format ``<language>/<document_name>.<page_number>[_index].jpg``.
        annotations: List of :class:`PPOCRLabelAnnotation` objects for this page.
    """

    image_path: str
    annotations: list[PPOCRLabelAnnotation]

    @field_validator("image_path")
    @classmethod
    def _validate_image_path(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("image_path must be non-empty")
        return v.strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bbox_to_points(bbox: tuple[int, int, int, int]) -> list[list[int]]:
    """Convert an axis-aligned bounding box to four ordered corner points.

    The point order is: top-left → top-right → bottom-right → bottom-left.

    Args:
        bbox: ``(x_min, y_min, x_max, y_max)`` bounding box tuple.

    Returns:
        A list of four ``[x, y]`` integer pairs.

    Example:
        >>> bbox_to_points((10, 20, 30, 40))
        [[10, 20], [30, 20], [30, 40], [10, 40]]
    """
    x0, y0, x1, y1 = bbox
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def make_image_path(
    document_name: str,
    language: str,
    page_number: int,
    index: int | None = None,
) -> str:
    """Build the Label.txt image path for a given page.

    The format matches the on-disk file structure
    ``<document_name>/<language>/<document_name>.<page_number:03d>[_index].jpg``
    but stores only the ``<language>/…`` portion so that the Label.txt is
    portable relative to the dataset root.

    Args:
        document_name: The document stem (e.g. ``"my_doc"``).
        language: The language folder (e.g. ``"vi"`` or ``"ch"``).
        page_number: 1-indexed document page number.
        index: Optional image sub-index (used when a PDF page yields multiple
            images).  When ``None`` the ``_index`` suffix is omitted.

    Returns:
        A relative path string such as ``"vi/my_doc.001.jpg"`` or
        ``"vi/my_doc.001_2.jpg"``.

    Example:
        >>> make_image_path("doc", "vi", 1)
        'vi/doc.001.jpg'
        >>> make_image_path("doc", "vi", 1, index=2)
        'vi/doc.001_2.jpg'
    """
    stem = f"{document_name}.{page_number:03d}"
    if index is not None:
        stem = f"{stem}_{index}"
    return f"{language}/{stem}.jpg"


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------


def ocr_page_to_ppocr(
    page: OCRPage,
    image_path: str,
    *,
    document_page_number: int | None = None,
    align_to: int | None = None,
    skip_first_paragraph: bool = False,
) -> PPOCRLabelPage:
    """Convert an :class:`OCRPage` to a :class:`PPOCRLabelPage`.

    Conversion rules:

    - Use ``OCRParagraph.text`` as transcription (NFC-normalized, embedded
      newlines replaced with spaces).
    - Use ``OCRParagraph.bbox`` as the bounding box source.  Paragraphs with
      no usable bounding box are **skipped** (not silently corrupted).
    - Convert ``(x_min, y_min, x_max, y_max)`` to four points via
      :func:`bbox_to_points`.
    - Paragraph order is preserved.
    - OCR-only fields (confidence, word-level data) are not present on
      :class:`OCRParagraph` after the schema migration and are therefore
      naturally excluded.
    - ``document_page_number`` is attached to every annotation when provided
      (defaulting to ``page.document_page_number`` if not explicitly passed).
    - ``align_to`` is attached to every annotation when provided.
    - If ``skip_first_paragraph`` is True, the first paragraph (header) and its
      bounding box are ignored.

    Args:
        page: Source :class:`OCRPage`.
        image_path: Relative image path for the Label.txt record.
        document_page_number: Optional document page number to attach to annotations.
            If None, uses ``page.document_page_number``.
        align_to: Optional page number to attach to every annotation for the
            later alignment stage.
        skip_first_paragraph: Whether to omit the first paragraph (e.g. header).

    Returns:
        A :class:`PPOCRLabelPage` ready for serialization.
    """
    doc_page_num = (
        document_page_number
        if document_page_number is not None
        else getattr(page, "document_page_number", None)
    )
    paragraphs = page.paragraphs
    if skip_first_paragraph and paragraphs:
        paragraphs = paragraphs[1:]

    annotations: list[PPOCRLabelAnnotation] = []
    for para in paragraphs:
        if para.bbox is None:
            continue
        transcription = (
            unicodedata.normalize("NFC", para.text).replace("\n", " ").strip()
        )
        if not transcription:
            continue
        annotations.append(
            PPOCRLabelAnnotation(
                transcription=transcription,
                points=bbox_to_points(para.bbox),
                document_page_number=doc_page_num,
                align_to=align_to,
            )
        )
    return PPOCRLabelPage(image_path=image_path, annotations=annotations)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_label_txt(
    pages: list[PPOCRLabelPage],
    output_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write a list of PPOCRLabel page entries to a ``Label.txt`` file.

    Each line in the output file has the form::

        <image_path>\\t<JSON annotations>\\n

    Requirements:

    - UTF-8 encoding with ``ensure_ascii=False``.
    - One physical line per image (transcriptions must not contain embedded
      newlines — the converter already guarantees this).
    - Pages are written in the order supplied by the caller.
    - ``align_to`` fields with value ``None`` are excluded from the serialized
      JSON to keep the output clean.

    Args:
        pages: Ordered list of :class:`PPOCRLabelPage` entries to export.
        output_path: Destination file path (e.g. ``Path("output/Label.txt")``).
        overwrite: When ``False`` (default) raise :exc:`FileExistsError` if the
            file already exists.  Pass ``True`` to overwrite silently.

    Raises:
        FileExistsError: If *output_path* exists and *overwrite* is ``False``.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Label.txt already exists at '{output_path}'. "
            "Pass overwrite=True to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fh:
        for page in pages:
            annotations_data = []
            for ann in page.annotations:
                entry: dict[str, Any] = {
                    "transcription": ann.transcription,
                    "points": ann.points,
                }
                if ann.document_page_number is not None:
                    entry["document_page_number"] = ann.document_page_number
                if ann.align_to is not None:
                    entry["align_to"] = ann.align_to
                annotations_data.append(entry)
            json_str = json.dumps(annotations_data, ensure_ascii=False)
            fh.write(f"{page.image_path}\t{json_str}\n")


def write_file_state_txt(
    pages: list[PPOCRLabelPage],
    output_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write a ``fileState.txt`` marking all supplied pages as completed (``1``).

    Each line has the form::

        <image_path>\\t1\\n

    The ordering is consistent with :func:`write_label_txt` for the same page list.

    Args:
        pages: Ordered list of :class:`PPOCRLabelPage` entries that are completed.
        output_path: Destination file path (e.g. ``Path("output/fileState.txt")``).
        overwrite: When ``False`` (default) raise :exc:`FileExistsError` if the
            file already exists.  Pass ``True`` to overwrite silently.

    Raises:
        FileExistsError: If *output_path* exists and *overwrite* is ``False``.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"fileState.txt already exists at '{output_path}'. "
            "Pass overwrite=True to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fh:
        for page in pages:
            fh.write(f"{page.image_path}\t1\n")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse_label_txt(path: Path) -> list[PPOCRLabelPage]:
    """Parse an existing ``Label.txt`` file into :class:`PPOCRLabelPage` objects.

    Each record is a tab-separated line::

        <image_path>\\t<JSON annotations array>

    Validation applied per record:

    - The line must split into exactly two parts on the first tab.
    - The image path must be non-empty.
    - The JSON payload must parse as a list.
    - Each annotation must have a non-empty ``transcription`` string.
    - Each annotation must have exactly four ``[x, y]`` integer coordinate pairs.

    Args:
        path: Path to the ``Label.txt`` file.

    Returns:
        List of :class:`PPOCRLabelPage` objects, one per valid line.

    Raises:
        ValueError: On any malformed record, with the 1-indexed source line
            number included in the message.  Malformed records are never
            silently discarded.
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Label.txt not found: '{path}'")

    pages: list[PPOCRLabelPage] = []
    raw_lines = path.read_text(encoding="utf-8").splitlines()

    for lineno, raw_line in enumerate(raw_lines, start=1):
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue

        parts = line.split("\t", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Label.txt line {lineno}: expected '<image_path>\\t<json>', "
                f"got {line!r}"
            )

        image_path_str, json_str = parts
        image_path_str = image_path_str.strip()
        if not image_path_str:
            raise ValueError(f"Label.txt line {lineno}: image path is empty")

        try:
            raw_annotations = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Label.txt line {lineno}: invalid JSON — {exc}"
            ) from exc

        if not isinstance(raw_annotations, list):
            raise ValueError(
                f"Label.txt line {lineno}: JSON payload must be a list, "
                f"got {type(raw_annotations).__name__}"
            )

        annotations: list[PPOCRLabelAnnotation] = []
        for ann_idx, raw_ann in enumerate(raw_annotations):
            if not isinstance(raw_ann, dict):
                raise ValueError(
                    f"Label.txt line {lineno}, annotation {ann_idx}: "
                    f"expected a dict, got {type(raw_ann).__name__}"
                )

            # Validate transcription
            transcription = raw_ann.get("transcription", "")
            if not isinstance(transcription, str) or not transcription.strip():
                raise ValueError(
                    f"Label.txt line {lineno}, annotation {ann_idx}: "
                    "transcription must be a non-empty string"
                )

            # Validate points
            raw_points = raw_ann.get("points", [])
            if not isinstance(raw_points, list) or len(raw_points) != 4:
                raise ValueError(
                    f"Label.txt line {lineno}, annotation {ann_idx}: "
                    f"points must contain exactly 4 pairs, got {len(raw_points) if isinstance(raw_points, list) else raw_points!r}"
                )
            validated_points: list[list[int]] = []
            for pt_idx, pt in enumerate(raw_points):
                if not isinstance(pt, list) or len(pt) != 2:
                    raise ValueError(
                        f"Label.txt line {lineno}, annotation {ann_idx}, "
                        f"point {pt_idx}: expected [x, y], got {pt!r}"
                    )
                if not all(isinstance(c, int) for c in pt):
                    raise ValueError(
                        f"Label.txt line {lineno}, annotation {ann_idx}, "
                        f"point {pt_idx}: coordinates must be integers, got {pt!r}"
                    )
                validated_points.append([int(pt[0]), int(pt[1])])

            document_page_number = raw_ann.get("document_page_number")
            if document_page_number is not None and not isinstance(document_page_number, int):
                raise ValueError(
                    f"Label.txt line {lineno}, annotation {ann_idx}: "
                    f"document_page_number must be an integer or null, got {document_page_number!r}"
                )

            align_to = raw_ann.get("align_to")
            if align_to is not None and not isinstance(align_to, int):
                raise ValueError(
                    f"Label.txt line {lineno}, annotation {ann_idx}: "
                    f"align_to must be an integer or null, got {align_to!r}"
                )

            annotations.append(
                PPOCRLabelAnnotation(
                    transcription=transcription.strip(),
                    points=validated_points,
                    document_page_number=document_page_number,
                    align_to=align_to,
                )
            )

        pages.append(PPOCRLabelPage(image_path=image_path_str, annotations=annotations))

    return pages


# ---------------------------------------------------------------------------
# Migration helper
# ---------------------------------------------------------------------------


def migrate_ocr_document_json(source_path: Path, dest_path: Path | None = None) -> Path:
    """Migrate a legacy ``OCRDocument`` JSON file to a flat ``list[OCRPage]`` JSON.

    The old format had the shape::

        {
            "source": "...",
            "language": "...",
            "page_count": N,
            "width": ...,
            "height": ...,
            "pages": [ { OCRPage dict }, ... ]
        }

    The new intermediate format is simply::

        [ { OCRPage dict }, ... ]

    Args:
        source_path: Path to the old ``OCRDocument`` JSON file.
        dest_path: Destination path for the migrated file.  Defaults to
            overwriting *source_path* in place.

    Returns:
        The path where the migrated file was written.

    Raises:
        FileNotFoundError: If *source_path* does not exist.
        ValueError: If the file does not look like an OCRDocument.
    """
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source JSON not found: '{source_path}'")

    raw = json.loads(source_path.read_text(encoding="utf-8"))

    if isinstance(raw, list):
        # Already the new format; nothing to do.
        return source_path

    if not isinstance(raw, dict) or "pages" not in raw:
        raise ValueError(
            f"'{source_path}' does not appear to be an OCRDocument JSON "
            "(expected a dict with a 'pages' key)"
        )

    pages_data: list[dict] = raw["pages"]
    out = json.dumps(pages_data, ensure_ascii=False, indent=2)

    write_to = Path(dest_path) if dest_path is not None else source_path
    write_to.write_text(out, encoding="utf-8")
    return write_to


def parse_header_paragraph(
    text: str,
    language: str = "vi",
) -> tuple[int | None, int | None]:
    """Parse header paragraph for document page number and align-to page number.

    The header text is expected to contain at most 2 numbers formatted like:
    ``- <document_page> - [( <page_it_match_to> )]``.

    Rules:
    - First number is the document page number.
    - Second number (if present and language == "vi") is the align_to page number.
    - If language != "vi" (e.g. "ch" for sinonom) or no second number is found,
      align_to is None.

    Args:
        text: Header paragraph text.
        language: Document language ("vi" for Vietnamese, "ch" for Sino-Nom, etc.).

    Returns:
        A tuple of (document_page_number, align_to).
    """
    cleaned = unicodedata.normalize("NFC", text).strip()
    # Try matching standard hyphen pattern: - <doc_page> - [ ( <align_to> ) ]
    m = re.search(r"-\s*(\d+)\s*-(?:\s*\(\s*(\d+)\s*\))?", cleaned)
    if m:
        doc_page = int(m.group(1))
        align_to = int(m.group(2)) if m.group(2) else None
    else:
        # Fallback: find all integer sequences in text
        nums = re.findall(r"\d+", cleaned)
        if len(nums) == 1:
            doc_page = int(nums[0])
            align_to = None
        elif len(nums) >= 2:
            doc_page = int(nums[0])
            align_to = int(nums[1])
        else:
            return None, None

    if language != "vi":
        align_to = None

    return doc_page, align_to


__all__ = [
    "PPOCRLabelAnnotation",
    "PPOCRLabelPage",
    "bbox_to_points",
    "make_image_path",
    "migrate_ocr_document_json",
    "ocr_page_to_ppocr",
    "parse_header_paragraph",
    "parse_label_txt",
    "write_file_state_txt",
    "write_label_txt",
]
