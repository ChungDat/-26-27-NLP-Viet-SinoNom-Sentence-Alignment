from __future__ import annotations

from pydantic import BaseModel, Field


class OCRParagraph(BaseModel):
    paragraph_id: int
    bbox: tuple[int, int, int, int] | None = None  # (x_min, y_min, x_max, y_max)
    text: str


class OCRPage(BaseModel):
    pdf_page_number: int  # page number on pdf reader
    document_page_number: int  # actual page number on document
    page_name: str
    method: str
    paragraphs: list[OCRParagraph] = Field(default_factory=list)


__all__ = [
    "OCRPage",
    "OCRParagraph",
]
