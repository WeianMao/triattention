import csv
import json

def csv_to_jsonl(csv_file_path, jsonl_file_path):
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file, \
         open(jsonl_file_path, mode='w', encoding='utf-8') as jsonl_file:
        
        # 读取CSV文件
        csv_reader = csv.DictReader(csv_file)
        
        for row in csv_reader:
            # 构建JSON对象
            json_obj = {
                "question": row["Question"],
                "answer": row["Answer"],
                "cot": "",  # 这里留空，因为原始CSV中没有推理过程
                "id": f"{row['ID']}-{row['Problem Number']}"
            }
            
            # 写入JSONL文件
            jsonl_file.write(json.dumps(json_obj) + '\n')

if __name__ == "__main__":
    # 输入和输出文件路径
    input_csv = "/root/code/LongReasoning_zhl/datasets/aime/AIME_Dataset_1983_2024.csv"  # 替换为您的CSV文件路径
    output_jsonl = "/root/code/LongReasoning_zhl/datasets/aime/AIME_Dataset_1983_2024.jsonl"  # 输出JSONL文件路径
    
    # 执行转换
    csv_to_jsonl(input_csv, output_jsonl)
    print(f"转换完成，结果已保存到 {output_jsonl}")