import time
import numpy as np
import onnxruntime as ort

# ===== 参数 =====
model_path = "LLama3.2_1B_int8_ir9.onnx"     # 你的模型路径
profile_file = "llama3.2_1B_profile.json"
seq_len = 256
batch = 1
n_layers = 16                        # KV 层数（看报错里有 0..15）

# ===== 构造输入 =====
input_ids = np.array([[1] + [2]*(seq_len-1)], dtype=np.int64)
attention_mask = np.ones((batch, seq_len), dtype=np.int64)
position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

# past_key_values 的形状一般是 [batch, num_heads, past_len, head_dim]
# prefill 时 past_len=0 → 给空数组；某些导出图要求固定形状，可以给 [batch, n_heads, 0, head_dim]
# 这里假设允许空
kv_inputs = {}
for i in range(n_layers):
    kv_inputs[f"past_key_values.{i}.key"] = np.zeros((batch, 8, 0, 64), dtype=np.float32)
    kv_inputs[f"past_key_values.{i}.value"] = np.zeros((batch, 8, 0, 64), dtype=np.float32)

# ===== 创建 Session (单线程) =====
opt = ort.SessionOptions()
opt.intra_op_num_threads = 2
opt.inter_op_num_threads = 2
opt.enable_profiling = True
opt.profile_file_prefix = profile_file.replace(".json","")  # 会自动加 .json
session = ort.InferenceSession(model_path, sess_options=opt, providers=["CPUExecutionProvider"])

ort_inputs = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "position_ids": position_ids,
    **kv_inputs,
}

# ===== 预热 =====
_ = session.run(None, ort_inputs)

# ===== 性能测试 =====
N = 10
times = []
for i in range(N):
    t0 = time.perf_counter()
    _ = session.run(None, ort_inputs)
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)
    print(f"Run {i+1}: {times[-1]:.2f} ms")

print(f"\nPrefill seq_len={seq_len}: avg {np.mean(times):.2f} ms | "
      f"min {np.min(times):.2f} ms | max {np.max(times):.2f} ms")

profile_file_generated = session.end_profiling()
print(f"Profile saved to {profile_file_generated}")


