# import pytesseract Can not use it on my server Beacause I am not sudoers
import easyocr
import cv2

# Text extractor from letter box
def extract_text_from_bbox(image, bbox):
    ocr_model = easyocr.Reader(["en"])
    # # x, y, w, h format version
    x_center, y_center, w, h = map(int, bbox)

    x_min = max(0, x_center - w // 2)
    x_max = min(image.shape[1], x_center + w // 2)
    y_min = max(0, y_center - h // 2)  # 상단
    y_max = min(image.shape[0], y_center + h // 2)  # 하단

    # ✅ OCR ROI 조정 (Bounding Box의 상단 부분을 좀 더 포함)
    y_min = max(0, y_min - int(0.2 * h))  # 위쪽 20% 확장

    roi = image[y_min:y_max, x_min:x_max]
    # cv2.imwrite('roi.png', roi)
    results = ocr_model.readtext(roi)
    extracted_text = "".join([res[1].upper() for res in results])
    # Extract text using OCR
    # text = pytesseract.image_to_string(roi, config="--psm 7")
    # text = text.strip().upper()
    # return text
    return extracted_text
    
    
def classify_foot_side(ocr_text):
    """
    Based on result of OCR, Classify Whether left or right is.
    """
    if ocr_text in ["R", "RST"]:
        return "right"
    elif ocr_text in ["L", "LST"]:
        return "left"
    return "unknown"

def get_foot_side(image, bbox):
    text = extract_text_from_bbox(image, bbox)
    return classify_foot_side(text)