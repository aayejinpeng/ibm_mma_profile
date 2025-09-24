import onnxruntime as ort
import numpy as np
import time

# 模型路径
model_path = "bertsquad-12-int8.onnx"
profile_file = "bert_profile.json"

# 单线程设置
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 2      # 单线程执行运算
sess_options.inter_op_num_threads = 2      # 单线程调度
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # 串行执行
sess_options.enable_profiling = True
sess_options.profile_file_prefix = profile_file.replace(".json","")  # 会自动加 .json

# 创建 Session
session = ort.InferenceSession(model_path, sess_options)

# 构造输入，按模型要求的 shape
batch_size = 1
seq_len = 256
vocab_size = 30522  # 根据模型词表


# 构造占位输入
inputs = {
    "unique_ids_raw_output___9:0": np.zeros((batch_size,), dtype=np.int64),
    "segment_ids:0": np.zeros((batch_size, seq_len), dtype=np.int64),
    "input_mask:0": np.ones((batch_size, seq_len), dtype=np.int64),
    "input_ids:0": np.random.randint(0, 30522, size=(batch_size, seq_len), dtype=np.int64)
}
# 热身
for _ in range(3):
    _ = session.run(None, inputs)

# 性能测试
num_iters = 10
start = time.time()
for _ in range(num_iters):
    outputs = session.run(None, inputs)
end = time.time()

print(f"Average latency: {(end - start) / num_iters * 1000:.2f} ms")

profile_file_generated = session.end_profiling()
print(f"Profile saved to {profile_file_generated}")


