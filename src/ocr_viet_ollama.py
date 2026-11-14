from sympy import imageset
from ollama import chat
from pathlib import Path

PROMPT = """
You are an OCR (Optical Character Recognition) engine. Your task is to extract all visible text from the uploaded image as accurately and completely as possible.

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

image_bytes = Path("Dong_hang/test/0100.png").read_bytes()

response = chat(
    model="qwen3-vl:2b-instruct",
    messages=[
        {
            "role": "user",
            "content": PROMPT,
            "images": [image_bytes]
        }
    ],
    options={
        "temperature": 0.1,
        "top_p": 0.9,
        "num_ctx": 8192,
        "repeat_penalty": 1.15,
        "num_predict": 512,
    }
)

print(response.message.content)

print("Prompt tokens:", response.prompt_eval_count)
print("Generated tokens:", response.eval_count)
print("Generation duration:", response.eval_duration)