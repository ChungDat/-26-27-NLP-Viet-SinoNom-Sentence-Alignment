from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_detection_model_name="PP-OCRv4_mobile_det",
    text_recognition_model_name="PP-OCRv4_mobile_rec",
    lang="vi"
)

result = ocr.predict("Dong_hang/test/0/0019.jpg")

for res in result:
    res.print()
    res.save_to_img("Dong_hang/paddle_output")
    res.save_to_json("Dong_hang/paddle_json")