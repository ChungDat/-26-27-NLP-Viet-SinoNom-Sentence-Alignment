import importlib.util
from pathlib import Path

_current_dir = Path(__file__).parent


def _import_page(module_name: str, file_name: str):
    file_path = _current_dir / file_name
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


upload_image = _import_page("upload_image", "1_upload_image.py")
preprocess_image = _import_page("preprocess_image", "2_preprocess_image.py")
ocr_image = _import_page("ocr_image", "3_ocr_image.py")
postprocess_image = _import_page("postprocess_image", "4_postprocess_image.py")

__all__ = [
    "upload_image",
    "preprocess_image",
    "ocr_image",
    "postprocess_image",
]
