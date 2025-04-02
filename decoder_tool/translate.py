import asyncio
import re
from googletrans import Translator
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# NLTK 리소스가 설치되어 있지 않다면 다운로드

nltk.download('stopwords')
nltk.download('punkt_tab')

def remove_stopwords(text, lang="english"):
    english_stopwords = set(stopwords.words(lang))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in english_stopwords]
    return ' '.join(filtered_words)

async def translate_mixed_text_and_remove_stopwords(text, dest='en'):
    translator = Translator()
    # 간단한 정규식으로 문장 단위 분리 (마침표, 느낌표, 물음표 뒤 공백 기준)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    processed_sentences = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        detected = await translator.detect(sentence)
        # 한국어이면 번역, 아니면 그대로 사용
        if detected.lang == 'ko':
            translated_sentence = await translator.translate(sentence, src='ko', dest=dest)
            translated_sentence = translated_sentence.text
        else:
            translated_sentence = sentence
        # 최종 영어 문장에서 불용어 제거
        filtered_sentence = remove_stopwords(translated_sentence, lang='english')
        processed_sentences.append(filtered_sentence)
    return " ".join(processed_sentences)

# 사용 예시
mixed_text = "[Finding] Test [Conclusion] 포함된 bone에 이상 소견은 보이지 않음."
final_text = asyncio.run(translate_mixed_text_and_remove_stopwords(mixed_text))
print(final_text)