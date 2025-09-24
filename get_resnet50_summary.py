import json
import csv

input_json_file = "resnet50_profile_2025-09-24_01-24-41.json"
output_csv_file = "resnet50_profile_filtered.csv"

with open(input_json_file, "r") as f:
    data = json.load(f)

with open(output_csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "dur", "op_name", "input_type_shape", "output_type_shape"])
    
    for node in data:
        name = node.get("name", "")
        if name.endswith("_fence_before") or name.endswith("_fence_after"):
            continue
        
        dur = node.get("dur", 0)
        args = node.get("args", {})
        op_name = args.get("op_name", "")
        input_type_shape = json.dumps(args.get("input_type_shape", []))
        output_type_shape = json.dumps(args.get("output_type_shape", []))
        
        writer.writerow([name, dur, op_name, input_type_shape, output_type_shape])

print(f"CSV file: {output_csv_file}")


import pandas as pd

# 读取之前生成的 CSV
df = pd.read_csv("resnet50_profile_filtered.csv")

# 保留关心的列
df = df[["name", "dur", "op_name", "input_type_shape", "output_type_shape"]]

# dur 转为数值
df["dur"] = pd.to_numeric(df["dur"], errors="coerce")

# 增加一列记录首次出现的顺序
df["first_index"] = df.groupby("name").cumcount().eq(0).cumsum()

# 按 name 分组，取 dur 中位数，并保留第一行的其他信息
df_agg = df.groupby("name").agg({
    "dur": "median",
    "op_name": "first",
    "input_type_shape": "first",
    "output_type_shape": "first",
    "first_index": "first"
}).reset_index()

# 按首次出现顺序排序
df_agg = df_agg.sort_values("first_index").drop(columns="first_index")

# 保存结果
df_agg.to_csv("resnet50_profile_median_ordered.csv", index=False)

print("CSV file: resnet50_profile_median_ordered.csv")


import pandas as pd
import ast

# 读取 CSV
df = pd.read_csv("resnet50_profile_median_ordered.csv")

# 将 shape 字符串转换成列表
df["input_type_shape"] = df["input_type_shape"].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
df["output_type_shape"] = df["output_type_shape"].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

# 找到残差加点的索引
residual_indices = df[df['name'].str.contains("resnetv17_stage") & df['name'].str.contains("plus")].index.tolist()

conv_params = []

# 先处理第一个残差块前的卷积
if residual_indices:
    first_idx = residual_indices[0]
    pre_block_df = df.loc[:first_idx-1]
    conv_rows = pre_block_df[pre_block_df['name'].str.contains("QLinearConv_token")]
    if not conv_rows.empty:
        min_ic_oc_idx = (conv_rows.apply(
            lambda row: next((s[list(s.keys())[0]][-1] for s in row["input_type_shape"] if s), 0) +
                        next((s[list(s.keys())[0]][-1] for s in row["output_type_shape"] if s), 0),
            axis=1
        )).idxmin()
        
        # 找 IC + OC 最小的卷积作为 3x3，如果相同再选 dur 最大
        ic_oc_sum = conv_rows.apply(
            lambda row: (
                next((s[list(s.keys())[0]][-1] for s in row["input_type_shape"] if s), 0) +
                next((s[list(s.keys())[0]][-1] for s in row["output_type_shape"] if s), 0)
            ), axis=1
        )
        min_ic_oc = ic_oc_sum.min()

        # 筛选 IC+OC 最小的卷积行
        candidates = conv_rows[ic_oc_sum == min_ic_oc]

        # 在这些候选中选 dur 最大的
        min_ic_oc_idx = candidates['dur'].idxmax()
        for idx, row in conv_rows.iterrows():
            input_shape = next((s[list(s.keys())[0]] for s in row["input_type_shape"] if s), None)
            output_shape = next((s[list(s.keys())[0]] for s in row["output_type_shape"] if s), None)
            if input_shape and output_shape:
                IC = input_shape[-1]
                IH = input_shape[1]
                IW = input_shape[2]
                OC = output_shape[-1]
                OH = output_shape[1]
                OW = output_shape[2]
                KH = KW = 3 if idx == min_ic_oc_idx else 1
                conv_params.append({
                    "name": row["name"], "IC": IC, "OC": OC,
                    "IH": IH, "IW": IW, "KH": KH, "KW": KW,
                    "OH": OH, "OW": OW, "dur": row["dur"]
                })

# 遍历每个残差块区间
for i, start_idx in enumerate(residual_indices):
    end_idx = residual_indices[i + 1] if i + 1 < len(residual_indices) else len(df)
    block_df = df.loc[start_idx:end_idx-1]

    # 保留残差块本身
    residual_row = df.loc[start_idx]
    conv_params.append({
        "name": residual_row["name"],
        "IC": "", "OC": "", "IH": "", "IW": "",
        "KH": "", "KW": "", "OH": "", "OW": "", "dur": residual_row["dur"]
    })

    # 找到区间内的卷积层
    conv_rows = block_df[block_df['name'].str.contains("QLinearConv_token")]
    if conv_rows.empty:
        continue

    # 找 IC + OC 最小的卷积作为 3x3
    min_ic_oc_idx = (conv_rows.apply(
        lambda row: next((s[list(s.keys())[0]][-1] for s in row["input_type_shape"] if s), 0) +
                    next((s[list(s.keys())[0]][-1] for s in row["output_type_shape"] if s), 0),
        axis=1
    )).idxmin()
    
    # 找 IC + OC 最小的卷积作为 3x3，如果相同再选 dur 最大
    ic_oc_sum = conv_rows.apply(
        lambda row: (
            next((s[list(s.keys())[0]][-1] for s in row["input_type_shape"] if s), 0) +
            next((s[list(s.keys())[0]][-1] for s in row["output_type_shape"] if s), 0)
        ), axis=1
    )
    min_ic_oc = ic_oc_sum.min()

    # 筛选 IC+OC 最小的卷积行
    candidates = conv_rows[ic_oc_sum == min_ic_oc]

    # 在这些候选中选 dur 最大的
    min_ic_oc_idx = candidates['dur'].idxmax()

    for idx, row in conv_rows.iterrows():
        input_shape = next((s[list(s.keys())[0]] for s in row["input_type_shape"] if s), None)
        output_shape = next((s[list(s.keys())[0]] for s in row["output_type_shape"] if s), None)
        if input_shape and output_shape:
            IC = input_shape[-1]
            IH = input_shape[1]
            IW = input_shape[2]
            OC = output_shape[-1]
            OH = output_shape[1]
            OW = output_shape[2]
            KH = KW = 3 if idx == min_ic_oc_idx else 1
            conv_params.append({
                "name": row["name"], "IC": IC, "OC": OC,
                "IH": IH, "IW": IW, "KH": KH, "KW": KW,
                "OH": OH, "OW": OW, "dur": row["dur"]
            })

# 保存最终 CSV
final_df = pd.DataFrame(conv_params)
final_df.to_csv("resnet50_conv_with_residual_final.csv", index=False)
print("CSV file: resnet50_conv_with_residual_final.csv")

