import re
import unicodedata
import torch

def normalize_special_tokens(cfg, text):
    """
    Report 내 스페셜 토큰을 일관된 형식으로 변환하고 불필요한 특수 문자를 제거합니다.
    """
    # x000D와 유사한 패턴 제거

    is_meerkat = "meerkat" in cfg.DECODER.NAME

    text = re.sub(r"x000D", "", text)  # "x000D" 제거
    text = re.sub(r"_+", " ", text)  # 연속된 밑줄(_)을 공백으로 대체
    text = re.sub(r"\s+", " ", text).strip()  # 불필요한 공백 제거

    # 스페셜 토큰 정규화
    special_tokens = ["FINDING", "CONCLUSION", "DIAGNOSIS", "RECOMMEND", "RECOMMENDATION"]
    for token in special_tokens:
        text = re.sub(rf"\[\s*{token}\s*\]", f"[{token}]", text, flags=re.IGNORECASE)
    
    text = text.replace("[EOS]", "").strip()
    text = text.replace("[BOS]", "").strip()

    # text = "[BOS] " + text + " [EOS]"
    
    return text



# TokenizedReportCollateFn 클래스 정의
# 데이터 배치(batch)를 처리하고 토큰화된 보고서를 생성하는 데 사용됩니다.
class TokenizedReportCollateFn:
    def __init__(self, tokenizer, cfg, max_length=128, save_path=None):
        """
        초기화 메서드
        Args:
            tokenizer: 텍스트를 토큰화하는 데 사용되는 tokenizer 객체
            max_length: 토큰화된 텍스트의 최대 길이 (기본값: 128)
        """
        self.tokenizer = tokenizer  # 토큰화를 처리할 tokenizer
        self.max_length = max_length  # 토큰화된 텍스트의 최대 길이
        self.save_path = save_path  # 토큰화된 보고서를 저장할 경로
        self.eos_token = self.tokenizer.eos_token
        self.cfg = cfg

    def _clean_report(self, text):
        # Normalize and remove non-ASCII characters.
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        text = re.sub(r'\[\s*finding\s*\]', '[FINDING]', text, flags=re.IGNORECASE)
        text = re.sub(r'\[\s*conclusion\s*\]', '[CONCLUSION]', text, flags=re.IGNORECASE)
        text = re.sub(r'\[\s*diagnosis\s*\]', '[DIAGNOSIS]', text, flags=re.IGNORECASE)
        parts = re.split(r'\[\s*recommend(?:ation)?\s*\]', text, flags=re.IGNORECASE)
        text = parts[0]
        text = text.replace('_x000D_', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        if text and not text.endswith(self.eos_token):
            text += ' ' + self.eos_token

        cleaned = re.sub(r'\[\s*(FINDING|DIAGNOSIS|CONCLUSION)\s*\]', '', text, flags=re.IGNORECASE).strip()
        cleaned = cleaned.replace(self.eos_token, '').strip()

        sentences = [s.strip() for s in re.split(r'\.\s*', cleaned) if s.strip()]
        N = len(sentences)
        if N % 2 == 0 and N > 0:
            half = N // 2
            if all(sentences[i].lower() == sentences[i + half].lower() for i in range(half)):
                final_text = '. '.join(sentences[:half]) + '.'
            else:
                final_text = '. '.join(sentences) + '.'
        else:
            final_text = '. '.join(sentences) + '.'

        final_text = final_text.strip() + ' ' + self.eos_token
        return final_text

    def __call__(self, batch):
        """
        배치를 처리하는 메서드
        Args:
            batch: 리스트 형태로 전달되는 배치 데이터.
                   각 요소는 (image, patch_tensor, label, report) 튜플로 구성됩니다.
        Returns:
            dict: 처리된 배치 데이터를 포함하는 딕셔너리
        """
        # 배치 데이터에서 각각의 요소를 분리
        images, patch_tensors, labels, reports = zip(*batch)

        # 이미지 텐서를 하나로 스택(stack)하여 배치 텐서로 변환
        images = torch.stack(images)
        # patch_tensors를 하나로 스택(stack)하여 배치 텐서로 변환
        patch_tensors = torch.stack(patch_tensors)

        # cleaned_reports = [normalize_special_tokens(self.cfg, report) for report in reports]
        cleaned_reports = [self._clean_report(report) for report in reports]

        # 보고서를 토큰화(tokenization)하여 토큰 ID와 어텐션 마스크 생성
        tokenized_reports = self.tokenizer(
            cleaned_reports,  # 보고서 리스트
            padding="longest",  # 가장 긴 문장에 맞춰 패딩 추가
            truncation=True,  # max_length를 초과하는 부분은 잘라냄
            max_length=self.max_length,  # 토큰화된 문장의 최대 길이
            return_tensors="pt"  # PyTorch 텐서로 반환
        )

        labels_as_sequence = tokenized_reports["input_ids"].clone()  # `input_ids`를 복사
        labels_as_sequence[labels_as_sequence == self.tokenizer.pad_token_id] = -100  # 패딩 토큰을 -100으로 설정

        # 토큰화된 결과를 파일로 저장
        if self.save_path:
            with open(self.save_path, "a") as f:
                for report, input_ids in zip(cleaned_reports, tokenized_reports["input_ids"]):
                    f.write(f"Original Report: {report}\n")
                    f.write(f"Tokenized IDs: {input_ids.tolist()}\n")
                    f.write(f"Decoded Text: {self.tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)}\n")
                    f.write("=" * 50 + "\n")

        # 처리된 데이터를 딕셔너리 형태로 반환
        return {
            "images": images,  # 배치 이미지 텐서
            "patch_tensors": patch_tensors,  # 배치 patch 텐서
            "class_labels": torch.stack(labels),  # 라벨을 스택하여 텐서로 변환
            "input_ids": tokenized_reports["input_ids"],  # 토큰화된 입력 ID
            "attention_mask": tokenized_reports["attention_mask"],  # 어텐션 마스크
            "labels": labels_as_sequence
        }