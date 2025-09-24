import onnxruntime as ort
import numpy as np
import time
import json

model_path = "resnet_int8.onnx"
profile_file = "resnet50_profile.json"

# 创建 SessionOptions
so = ort.SessionOptions()
so.enable_profiling = True
so.profile_file_prefix = profile_file.replace(".json","")  # 会自动加 .json

# 设置线程数为 1
so.intra_op_num_threads = 2   # 单算子内部线程
so.inter_op_num_threads = 2   # 不同算子间线程

# 创建 session
sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])

# 查看输入信息
for inp in sess.get_inputs():
    print(f"Input name: {inp.name}, shape: {inp.shape}, type: {inp.type}")

input_name = sess.get_inputs()[0].name
input_shape = [dim if isinstance(dim, int) else 1 for dim in sess.get_inputs()[0].shape]

# 随机输入
x = np.random.rand(*input_shape).astype(np.float32)

# 预热
_ = sess.run(None, {input_name: x})

# 正式运行
N = 20
t0 = time.perf_counter()
for _ in range(N):
    _ = sess.run(None, {input_name: x})
t1 = time.perf_counter()

avg_ms = (t1 - t0) / N * 1000
fps = N / (t1 - t0)
print(f"Avg latency: {avg_ms:.2f} ms, FPS: {fps:.2f}")

# 结束 profiling 并生成文件
profile_file_generated = sess.end_profiling()
print(f"Profile saved to {profile_file_generated}")


