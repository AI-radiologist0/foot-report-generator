import os
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

# vis_graph 폴더 생성
output_dir = "vis_graph"
os.makedirs(output_dir, exist_ok=True)

# 피클 파일 로드
pkl_path = "data/pkl/output_report.pkl"
with open(pkl_path, 'rb') as f:
    report_data = pickle.load(f)

# 텍스트 데이터 추출
findings = []
conclusions = []
recommendations = []

for report in report_data.values():
    if report.get("finding"):
        findings.append(report["finding"])
    if report.get("conclusion"):
        conclusions.append(report["conclusion"])
    if report.get("recommend"):
        recommendations.append(report["recommend"])

# 리스트를 하나의 문자열로 합치기
finding_text = " ".join(findings)
conclusion_text = " ".join(conclusions)
recommend_text = " ".join(recommendations)


# 1. 워드클라우드 생성
def generate_wordcloud(text, title, filename):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


generate_wordcloud(finding_text, "Findings WordCloud", "findings_wordcloud.png")
generate_wordcloud(conclusion_text, "Conclusions WordCloud", "conclusions_wordcloud.png")
# generate_wordcloud(recommend_text, "Recommendations WordCloud", "recommendations_wordcloud.png")


# 2. 보고서 길이 분포 분석
def plot_length_distribution(texts, title, filename):
    lengths = [len(text.split()) for text in texts]
    plt.figure(figsize=(8, 5))
    sns.histplot(lengths, bins=30, kde=True)
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


plot_length_distribution(findings, "Findings Length Distribution", "findings_length_dist.png")
plot_length_distribution(conclusions, "Conclusions Length Distribution", "conclusions_length_dist.png")
plot_length_distribution(recommendations, "Recommendations Length Distribution", "recommendations_length_dist.png")


# 3. 등장한 모든 단어의 빈도 분석
def count_all_words(text):
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())  # 영어 단어만 추출
    return Counter(words)

finding_word_counts = count_all_words(finding_text)
conclusion_word_counts = count_all_words(conclusion_text)
recommend_word_counts = count_all_words(recommend_text)


# 상위 20개 단어 시각화
def plot_top_words(word_counts, title, filename):
    top_words = dict(word_counts.most_common(20))  # 상위 20개 단어 선택
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(top_words.keys()), y=list(top_words.values()))
    plt.xticks(rotation=45)
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


plot_top_words(finding_word_counts, "Top 20 Words in Findings", "findings_top_words.png")
plot_top_words(conclusion_word_counts, "Top 20 Words in Conclusions", "conclusions_top_words.png")
plot_top_words(recommend_word_counts, "Top 20 Words in Recommendations", "recommendations_top_words.png")

print(f"All visualizations saved in {output_dir}/")
