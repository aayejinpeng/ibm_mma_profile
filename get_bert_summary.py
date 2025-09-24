import json
import csv

json_file = "bert_profile_2025-09-24_01-23-43.json"
csv_file = "bert.csv"

fields = [
    "name", "op_name", "provider", "node_index", "dur",
    "activation_size", "output_size", "input_type_shape", "output_type_shape"
]

rows = []

with open(json_file, "r") as f:
    for line in f:
        line = line.strip().rstrip(",")  # 去掉末尾逗号
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            print(f"Skipping invalid line: {line[:50]}...")  # 打印前 50 个字符
            continue

        args = obj.get("args", {})
        row = {
            "name": obj.get("name", ""),
            "op_name": args.get("op_name", ""),
            "provider": args.get("provider", ""),
            "node_index": args.get("node_index", ""),
            "dur": obj.get("dur", ""),
            "activation_size": args.get("activation_size", ""),
            "output_size": args.get("output_size", ""),
            "input_type_shape": json.dumps(args.get("input_type_shape", "")),
            "output_type_shape": json.dumps(args.get("output_type_shape", "")),
        }
        rows.append(row)

with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"CSV saved to {csv_file}")
