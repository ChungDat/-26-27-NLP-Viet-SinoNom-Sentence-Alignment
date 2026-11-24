from __future__ import annotations

import logging
from pathlib import Path

import cv2
import sys
import numpy as np
import pymupdf
import pytesseract

logger = logging.getLogger(__name__)


class PDFDocument:
    """Manages PDF document loading and embedded image extraction."""

    def __init__(
        self,
        pdf_source: str | Path | bytes,
        orientation_threshold: float = 3.0,
    ) -> None:
        """Initialize PDFDocument with a source and orientation confidence threshold.

        Args:
            pdf_source: File path (str or Path) or raw bytes of the PDF.
            orientation_threshold: Minimum confidence score required to apply OSD rotation.

        Raises:
            FileNotFoundError: If pdf_source is a path that does not exist.
            ValueError: If pdf_source is empty bytes or orientation_threshold is negative.
        """
        if isinstance(pdf_source, (str, Path)):
            source_path = Path(pdf_source)
            if not source_path.is_file():
                raise FileNotFoundError(f"PDF file does not exist: {source_path}")
            self.pdf_source: str | Path | bytes = source_path
        elif isinstance(pdf_source, bytes):
            if len(pdf_source) == 0:
                raise ValueError("PDF source bytes cannot be empty.")
            self.pdf_source = pdf_source
        else:
            raise ValueError(
                f"Unsupported pdf_source type: {type(pdf_source).__name__}. "
                "Expected str, Path, or bytes."
            )

        if orientation_threshold < 0:
            raise ValueError("Orientation threshold must be non-negative.")
        self.orientation_threshold = orientation_threshold

    def open_pdf(self) -> pymupdf.Document:
        """Open a PDF document.

        Returns:
            A PyMuPDF Document instance.

        Raises:
            ValueError: If the PDF cannot be opened or is corrupted.
        """
        try:
            if isinstance(self.pdf_source, bytes):
                return pymupdf.open(stream=self.pdf_source, filetype="pdf")
            return pymupdf.open(self.pdf_source)
        except Exception as error:
            raise ValueError(f"Could not open PDF document: {error}") from error

    def page_count(self) -> int:
        """Return the total number of pages in the PDF document.

        Returns:
            The total page count.

        Raises:
            ValueError: If opening the PDF document fails.
        """
        with self.open_pdf() as document:
            return len(document)

    @classmethod
    def rotate_image(
        cls,
        image: np.ndarray,
        orientation_threshold: float,
        language: str = "vi",
    ) -> tuple[np.ndarray, bool]:
        """Rotate image according to orientation detected by Tesseract OSD.

        Args:
            image: Input image as a NumPy ndarray.
            orientation_threshold: Confidence threshold to apply rotation.
            language: Language code ('vi' for Vietnamese, 'ch' for Sino-Nom/Chinese).

        Returns:
            A tuple of (processed_image, was_rotated) where processed_image is the
            NumPy ndarray and was_rotated indicates whether rotation was applied.

        Raises:
            ValueError: If language is not 'vi' or 'ch'.
        """
        if language == "vi":
            tess_lang = "vie"
        elif language == "ch":
            tess_lang = "chi_tra"
        else:
            raise ValueError(
                f"Unsupported language: {language!r}. Language must be 'vi' or 'ch'."
            )

        img = image.copy()
        try:
            orientation_data = pytesseract.image_to_osd(
                img, output_type=pytesseract.Output.DICT, lang=tess_lang
            )
        except (pytesseract.TesseractError, Exception) as error:
            logger.warning("OSD failed (%s). Keeping original orientation.", error)
            return img, False

        confidence = orientation_data.get("orientation_conf", 0.0)
        rotate_angle = orientation_data.get("rotate", 0)

        if confidence >= orientation_threshold:
            if rotate_angle == 90:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                return img, True
            if rotate_angle == 180:
                img = cv2.rotate(img, cv2.ROTATE_180)
                return img, True
            if rotate_angle == 270:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                return img, True

        return img, False

    def save_images(
        self,
        pdf_pages: list[int],
        document_name: str | Path,
        document_type: str = "vi",
    ) -> None:
        """Extract embedded images from selected pages and save to disk in PPOCRLabel format.

        Selected pages are 1-indexed. The images are saved under the directory
        `Path(document_name) / document_type` as JPEG (.jpg) files named:
        `<stem>.<page_number:03d>.jpg` (or `<stem>.<page_number:03d>_<img_index>.jpg`
        if multiple embedded images exist on the page).

        Args:
            pdf_pages: List of 1-indexed page numbers to extract.
            document_name: Document name or base path for output files.
            document_type: Document type folder name ('vi', 'ch' or 'extra').

        Returns:
            None.

        Raises:
            ValueError: If pdf_pages is empty, any page number is out of bounds,
                or document_type is unsupported.
            RuntimeError: If a requested page contains no embedded images or image
                decoding/saving fails.
        """
        if document_type not in ("vi", "ch", "extra"):
            raise ValueError(
                f"Unsupported document type: {document_type!r}. Language must be 'vi', 'ch' or 'extra'."
            )

        if not pdf_pages:
            raise ValueError("pdf_pages list cannot be empty.")

        doc_stem = (
            Path(document_name).stem
            if Path(document_name).suffix
            else Path(document_name).name
        )
        destination_path = Path(document_name) / document_type
        destination_path.mkdir(parents=True, exist_ok=True)

        with self.open_pdf() as document:
            total_pages = len(document)

            for page_num in pdf_pages:
                if not 1 <= page_num <= total_pages:
                    raise ValueError(
                        f"Page number {page_num} is out of bounds. "
                        f"Document has {total_pages} page(s)."
                    )

            for page_num in pdf_pages:
                page_index = page_num - 1
                page = document[page_index]
                page_images = page.get_images()

                if not page_images:
                    raise RuntimeError(f"Page {page_num} contains no embedded images.")

                for img_index, img_info in enumerate(page_images):
                    xref = img_info[0]
                    base_image = document.extract_image(xref)
                    img_bytes = base_image.get("image")
                    if not img_bytes:
                        raise RuntimeError(
                            f"Could not extract image data for page {page_num} (xref: {xref})."
                        )

                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        raise RuntimeError(
                            f"Failed to decode image data for page {page_num} (xref: {xref})."
                        )

                    if document_type != "extra":
                        img, _ = self.rotate_image(
                            img, self.orientation_threshold, document_type
                        )

                    # Ensure standard 3-channel BGR for JPEG saving
                    if len(img.shape) == 3 and img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                    if len(page_images) == 1:
                        file_name = f"{doc_stem}.{page_num:03d}.jpg"
                    else:
                        file_name = f"{doc_stem}.{page_num:03d}_{img_index}.jpg"

                    output_file = destination_path / file_name
                    write_success = cv2.imwrite(str(output_file), img)
                    if not write_success:
                        raise RuntimeError(
                            f"Failed to save image to disk: {output_file}"
                        )


def select_pages(start: int, end: int, step: int = 1) -> list[int]:
    """Create an ordered list of 1-indexed page numbers.

    Args:
        start: The first page number (1-indexed).
        end: The last page number (inclusive, 1-indexed).
        step: The step increment between pages (must be >= 1).

    Returns:
        A list of 1-indexed page numbers between start and end.

    Raises:
        ValueError: If start < 1, end < 1, or step < 1.
    """
    if start < 1 or end < 1:
        raise ValueError(
            f"Page numbers must be >= 1. Received start={start}, end={end}."
        )
    if step < 1:
        raise ValueError(f"Step must be >= 1. Received step={step}.")

    if start <= end:
        return list(range(start, end + 1, step))
    return list(range(start, end - 1, -step))


def main() -> None:
    """Extract sample pages from a PDF in each supported image mode."""

    pdf_source = Path(
        "CV19-Phạm Phúc An Khang-20120303/T1_Sách Truyện Các Thánh (Tháng 1).pdf"
    )
    document_name = Path("T1_Sách Truyện Các Thánh (Tháng 1)")
    pdf_pages = select_pages(450, 458, 1)
    orientation_threshold = 3.0

    pdf_document = PDFDocument(pdf_source, orientation_threshold)
    pdf_document.save_images(pdf_pages, document_name, "ch")


if __name__ == "__main__":
    main()
