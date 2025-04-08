import re
import pickle
from tqdm import tqdm
from googletrans import Translator

# 텍스트 클린 함수
def clean_text(text):
    if text:
        text = re.sub(r'[\W]?x000D[\W]?', ' ', text)  # 'x000D'와 주변 특수문자 제거
        text = re.sub(r'[_]', ' ', text)  # '_' 제거
        text = re.sub(r'[\r\n]+', ' ', text)  # 개행 문자 (\r, \n) 제거
        text = re.sub(r'\s+', ' ', text).strip()  # 연속된 공백 제거 및 앞뒤 공백 제거
    return text

# 번역 함수
translator = Translator()

def translate_text(text, src='ko', dest='en'):
    try:
        if re.search(r'[가-힣]', text):  # 한글 포함 여부 확인
            translated = translator.translate(text, src=src, dest=dest)
            return translated.text
    except Exception as e:
        print(f"Translation failed: {e}")
    return text  # 번역 실패 시 원본 텍스트 반환

# 텍스트 처리 함수 (클린 + 번역)
def process_text(text):
    cleaned_text = clean_text(text)  # 텍스트 클린
    return translate_text(cleaned_text, src='ko', dest='en')  # 한국어 번역

# 키워드 기반 섹션 매칭
keywords = {
    'finding': ['finding'],
    'conclusion': ['conclusion', 'diagnosis'],
    'recommend': ['recommend', 'recommendation']
}

def match_section(section_name):
    for key, keyword_list in keywords.items():
        if any(keyword in section_name.lower() for keyword in keyword_list):
            return key
    return None  # 매칭되지 않는 경우

# 최종 데이터 처리 및 저장
def process_and_save(data, report_pkl_path):
    final_report_info = {}

    for id, value in tqdm(data.items()):
        # 기본 정보 가져오기
        image_path = data[id].get('file_path', None)
        patient_id = data[id].get('patient_id', None)

        disease_classification = data[id].get('class', None)
        bbx = data[id].get('bbx', None)
        image_id = data[id].get('image_id', None)
        original_report = data[id].get('diagnosis', None)

        # 섹션 이름 탐지
        matches = re.findall(r'\[(.*?)\]', original_report, re.IGNORECASE) if original_report else []
        sections = {match_section(match): match for match in matches if match_section(match)}

        # 섹션별 데이터 추출 및 재정렬
        reordered_sections = {'finding': None, 'conclusion': None, 'recommend': None}

        for section_key, section_name in sections.items():
            if section_key:
                pattern = rf'\[{section_name}\s*\](.*?)(?=\[|$)'  # 다음 섹션 또는 끝까지
                match = re.search(pattern, original_report, re.IGNORECASE | re.DOTALL)
                if match:
                    raw_text = match.group(1)
                    processed_text = process_text(raw_text)  # 클린 및 번역 처리
                    reordered_sections[section_key] = processed_text

        # 재정렬된 보고서 생성 (공백 하나로 구분)
        reordered_report = (
            f"[FINDING] {reordered_sections['finding']} "
            f"\n[CONCLUSION] {reordered_sections['conclusion']} "
            f"\n[RECOMMEND] {reordered_sections['recommend']}"
        ).strip()

        # 최종 데이터 저장
        final_report_info[id] = {
            'patient_id': patient_id,
            'file_path': image_path,
            'image_id': image_id,
            'bbx': bbx,
            'class': disease_classification,
            'original_diagnosis': original_report,
            'diagnosis': reordered_report
        }

    # ID 순서대로 정렬
    final_report_info = dict(sorted(final_report_info.items(), key=lambda x: x[0]))

    # 최종 데이터 저장
    with open(report_pkl_path, 'wb') as pkl_file:
        pickle.dump(final_report_info, pkl_file)

    print(f"Final data saved to {report_pkl_path}")


if __name__ == "__main__":
    # 데이터 로드
    with open('data/pkl/output200x300.pkl', 'rb') as pkl_file:
        data = pickle.load(pkl_file)
        print("Data loaded successfully.")
    # 최종 데이터 처리 및 저장
    process_and_save(data, 'data/pkl/output_report(eng).pkl')