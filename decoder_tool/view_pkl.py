import pickle

# with open('data/pkl/output200x300.pkl', 'rb') as f:
#     data = pickle.load(f)

with open('data/pkl/output_report(eng).pkl', 'rb') as f:
    data = pickle.load(f)

for key, value in data.items():
    # print(key, value)
    print(f"{key}, \n {value['file_path']} | {value['original_diagnosis']} \n")
    print("diagnosis: ", value['diagnosis'])
    print("============================================================")