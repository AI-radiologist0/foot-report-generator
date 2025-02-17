import os, json

def fill_diagnoses_from_other_json(output_data, diagnosis_json_path):
    """진단 정보가 있는 JSON 파일에서 정보를 가져와 output_data에 채워넣음"""
    
    # ✅ 1. 진단 정보가 있는 JSON 파일 로드
    with open(diagnosis_json_path, "r") as f:
        diagnosis_data = json.load(f)

    # ✅ 2. {patient_id: diagnosis} 매핑 생성
    patient_diagnoses = {}

    for record in diagnosis_data["data"]:
        patient_id = record["patient_id"]
        diagnosis = record.get("diagnosis", "").strip()

        # ✅ diagnosis가 있는 경우 저장
        if diagnosis:
            patient_diagnoses[patient_id] = diagnosis

    # ✅ 3. output_data 내의 diagnosis를 채움
    for record in output_data["data"]:
        patient_id = record["patient_id"]

        # ✅ 만약 diagnosis가 없으면, 다른 JSON에서 가져오기
        if not record.get("diagnosis", "").strip():
            record["diagnosis"] = patient_diagnoses.get(patient_id, "No diagnosis available")

    return output_data

# ✅ JSON 파일 경로 설정
diagnosis_json_path = "data/json/foot_merge.json"  # 진단 정보가 포함된 JSON
output_json_path = "data/json/foot_merge_filtered_by_yolo.json"  # 진단 정보가 없는 JSON

# ✅ 진단 정보가 없는 JSON 로드
with open(output_json_path, "r") as f:
    output_data = json.load(f)

# ✅ 진단 정보 업데이트
output_data = fill_diagnoses_from_other_json(output_data, diagnosis_json_path)

# ✅ 수정된 JSON 저장
with open(output_json_path, "w") as f:
    json.dump(output_data, f, indent=4)
print(f"✅ Updated JSON saved: {output_json_path}")
