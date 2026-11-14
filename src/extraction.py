from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytesseract


class PDFDocument:
    def __init__(
        self,
        pdf_source: str | Path | bytes,
        dpi: int,
        orientation_threshold: float,
    ):
        self.pdf_source = pdf_source
        self.dpi = dpi
        self.orientation_threshold = orientation_threshold

    def open_pdf(self) -> pymupdf.Document:
        """Open a PDF file."""

        if isinstance(self.pdf_source, bytes):
            return pymupdf.open(stream=self.pdf_source, filetype="pdf")
        return pymupdf.open(self.pdf_source)

    def page_count(self) -> int:
        """Return the number of pages in an uploaded PDF."""

        with self.open_pdf() as document:
            return len(document)

    def iter_images(self, pdf_pages: list[int]) -> Iterator[np.ndarray]:
        """Convert selected PDF pages into numpy ndarray images (in RGB format) iteratively.

        Args:
            pdf_pages: List of 1-indexed page numbers to extract.

        Returns:
            An extracted image.
        """
        with self.open_pdf() as document:
            for page_number in pdf_pages:
                if not 1 <= page_number <= len(document):
                    print(f"Skip invalid page {page_number}")
                    continue

                page = document[page_number - 1]
                pixmap = page.get_pixmap(dpi=self.dpi, alpha=False)

                image = np.frombuffer(
                    pixmap.samples,
                    dtype=np.uint8,
                ).reshape(pixmap.height, pixmap.width, pixmap.n)

                orientation_data = pytesseract.image_to_osd(
                    image,
                    output_type=pytesseract.Output.DICT,
                )

                if orientation_data["orientation_conf"] < self.orientation_threshold:
                    print(
                        f"Page {page_number} has orientation confidence lower than "
                        f"{self.orientation_threshold}"
                    )
                    continue

                if orientation_data["rotate"] == 0:
                    pass
                elif orientation_data["rotate"] == 90:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                elif orientation_data["rotate"] == 180:
                    image = cv2.rotate(image, cv2.ROTATE_180)
                elif orientation_data["rotate"] == 270:
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                else:
                    print(f"Page {page_number} has unusual orientation")

                yield image

    def to_images(
        self,
        pdf_pages: list[int],
    ) -> list[np.ndarray]:
        """Convert selected PDF pages into numpy ndarray images (in RGB format).

        Args:
            pdf_pages: List of 1-indexed page numbers to extract.

        Returns:
            A list of extracted images.
        """

        return list(self.iter_images(pdf_pages))

    def save_images(
        self,
        images: list[np.ndarray],
        pdf_pages: list[int],
        destination_path: str | Path,
    ) -> None:
        """Save extracted page images to a directory.

        Args:
            images: Images to save.
            pages: List of 1-indexed page numbers corresponding to ``images``.
            destination_path: Path to the destination directory.
        """

        destination = Path(destination_path)
        destination.mkdir(parents=True, exist_ok=True)

        for image, page_number in zip(images, pdf_pages):
            file_name = f"{page_number:04d}.png"
            cv2.imwrite(
                str(destination / file_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            )


def select_pages(start: int, end: int, step: int = 1) -> list[int]:
    """Create a list of selected page numbers.

    Args:
        start: The first page to select.
        end: The last page to select, inclusive.
        step: The number of pages to skip between selections.

    Returns:
        A list of 1-indexed page numbers between ``start`` and ``end``.
    """

    if start <= end:
        pages = list(range(start, end + 1, step))
    else:
        pages = list(range(start, end - 1, -step))
    return pages


def main() -> None:
    """Extract sample pages from a PDF in each supported image mode."""

    pdf_source = Path(
        "CV19-Phạm Phúc An Khang-20120303/T1_Sách Truyện Các Thánh (Tháng 1).pdf"
    )
    pdf_pages = select_pages(100, 100, 1)
    dpi = 300
    orientation_threshold = 3
    destination_path = Path("Dong_hang/test/")

    pdf_document = PDFDocument(pdf_source, dpi, orientation_threshold)
    images = pdf_document.to_images(pdf_pages)
    pdf_document.save_images(images, pdf_pages, destination_path)


if __name__ == "__main__":
    main()
