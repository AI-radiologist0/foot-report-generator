import pytesseract

# Text extractor from letter box
def extract_text_from_bbox(image, bbox):
    
    # # x, y, w, h format version
    x, y, w, h = map(int, bbox)
    # x1, y1, x2, y2 format version
    # x1, y1, x2, y2 = map(int, bbox)
    roi = image[y:y+h, x+x+w]
    
    # Extract text using OCR
    text = pytesseract.image_to_string(roi, config="--psm 7")
    text = text.strip().upper()
    return text

def classify_foot_side(ocr_text):
    """
    Based on result of OCR, Classify Whether left or right is.
    """
    if ocr_text in ["R", "RST"]:
        return "right"
    elif ocr_text in ["L", "LST"]:
        return "left"
    return "Unknown"

def get_foot_side(image, bbox):
    text = extract_text_from_bbox(image, bbox)
    return classify_foot_side(text)