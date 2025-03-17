import pickle
import os, re
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from collections import defaultdict

import _init_path
from config import cfg, update_config

# Data loading code
with open('data/pkl/output200x300.pkl', 'rb') as f:
    data = pickle.load(f)

report_pkl_path = os.path.join(os.path.dirname('data/pkl/output200x300.pkl'), 'cleaned_report_for_decoder_train.pkl')
print(report_pkl_path)

# check the number of report
cnt_diag = len(data)
print(f"Total Diagnosis Report Count: {cnt_diag}")

# Counts of each labels.

finding_pattern = None
conclusion_pattern = None
recommendation_pattern = None

report_info = defaultdict(dict)


def clean_text(text):
    if text:
        text = re.sub(r'[\W]?x000D[\W]?', ' ', text)  # 'x000D'와 주변 특수문자 제거
        text = re.sub(r'[_]', ' ', text)  # '_' 제거
        text = re.sub(r'[\r\n]+', ' ', text)  # 개행 문자 (\r, \n) 제거
        text = re.sub(r'\s+', ' ', text).strip()  # 연속된 공백 제거 및 앞뒤 공백 제거
    return text

# diagnosis (pkl_data["diagnosis"])
for id, value in tqdm(data.items()):
    sections = {
        'finding': None,
        'conclusion': None,
        'recommend': None,
    }

    image_path = data[id]['file_path']
    report = data[id]['diagnosis']

    report_type = os.path.dirname(image_path)
    report_type = 'nuri' if report_type.split('/')[4].lower().find("nuri") != -1 else 'old'

    if report_type == 'nuri':
        # cau 타입일 때 섹션 이름
        finding_pattern = r'\[finding\s*\](.*?)(?=\[conclusion\s*\])'
        conclusion_pattern = r'\[conclusion\s*\](.*?)(?=\[recommendation\s*\])'
        recommendation_pattern = r'\[recommendation\s*\](.*)'
    elif report_type == 'old':
        # old 타입일 때 섹션 이름
        finding_pattern = r'\[finding\s*\](.*?)(?=\[diagnosis\s*\])'
        conclusion_pattern = r'\[diagnosis\s*\](.*?)(?=\[recommend\s*\])'
        recommendation_pattern = r'\[recommend\s*\](.*)'

    finding_match = re.search(finding_pattern, report, re.IGNORECASE | re.DOTALL)
    conclusion_match = re.search(conclusion_pattern, report, re.IGNORECASE | re.DOTALL)
    recommendation_match = re.search(recommendation_pattern, report, re.IGNORECASE | re.DOTALL)

    if finding_match:
        sections['finding'] = clean_text(finding_match.group(1))
    if conclusion_match:
        sections['conclusion'] = clean_text(conclusion_match.group(1))
    if recommendation_match:
        sections['recommend'] = clean_text(recommendation_match.group(1))

    # Combine sections into a single report
    combined_report = (
        f"[FINDING] {sections['finding']} \n [CONCLUSION] {sections['conclusion']} \n [RECOMMEND] {sections['recommend']}"
    ).strip()

    # Add combined report to sections
    sections['diagnosis'] = combined_report

    for key, value in data[id].items():
        sections[key] = value

    report_info[id] = sections

# Save the processed data
with open(report_pkl_path, 'wb') as pkl_file:
    pickle.dump(report_info, pkl_file)

print(f"Final data saved to {report_pkl_path}")