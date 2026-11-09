import os
from src.ocr_viet import get_ocr_data, get_ocr_text, save_text, visualize, normalize_text
from src.metric import CER, WER

def run_config(configs, source_path, image_name):
    text_files = []

    for mode in range(0, 3):
        mode_path = os.path.join(source_path, str(mode))
        image_path = os.path.join(mode_path, image_name)

        for config in configs:
            data = get_ocr_data(image_path, config)
            text = get_ocr_text(image_path, config)
            text = normalize_text(text)
            visualize(image_path, data, mode_path, f"ocr_visualization {config}.jpg")
            text_name = f"{image_name}_text {config}.txt"
            save_text(text, mode_path, text_name)
            text_files.append(text_name)

    return text_files


def compare(hypothesis, reference):
    cer = CER(hypothesis, reference)
    wer = WER(hypothesis, reference)
    print(f"Reference file: {hypothesis}")
    print(f"CER: {cer * 100:.2f}% | WER: {wer * 100:.2f}%")

if __name__ == "__main__":
    configs = ["--psm 3", "--psm 6"]

    source_path = "Dong_hang/test/"
    dest_path = "Dong_hang/test/"
    image_name = "0019.jpg"
    
    text_files = run_config(configs, source_path, image_name)
    for file in text_files:
        for i in range(3):
            compare(os.path.join(source_path, str(i), file), os.path.join(source_path, "source.txt"))
            print()

    # compare("Dong_hang/test/source.txt", "Dong_hang/test/0/text --psm 3.txt")
