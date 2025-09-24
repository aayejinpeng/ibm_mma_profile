import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import numpy as np
import time
import pandas as pd

# 矩阵尺寸组合
records = []
repeat = 100
for K in range(64, 4096 + 1, 64):
    M = 256
    N = 256

    # 输入张量信息
    X = helper.make_tensor_value_info("X", TensorProto.INT8, [1, 256, K])
    W = helper.make_tensor_value_info("W", TensorProto.INT8, [K, 256])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT32, [1, 256, 256])

    # 创建 MatMulInteger 节点（int8 x int8 -> int32）
    matmul_node = helper.make_node(
        "MatMulInteger",
        inputs=["X", "W"],
        outputs=["Y"]
    )

    # 构建图
    graph_def = helper.make_graph(
        nodes=[matmul_node],
        name="Int8GEMM_graph",
        inputs=[X, W],
        outputs=[Y],
    )

    # 构建模型
    model_def = helper.make_model(
        graph_def,
        producer_name='int8_gemm_in_memory',
        ir_version=9,
        opset_imports=[onnx.helper.make_operatorsetid("", 19)]
    )

    # 创建 ONNXRuntime session（内存中）
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 2
    sess_options.inter_op_num_threads = 1
    sess = ort.InferenceSession(model_def.SerializeToString(), sess_options, providers=["CPUExecutionProvider"])

    # 构造随机 int8 输入
    X_data = np.random.randint(-128, 127, (1, N, K), dtype=np.int8)
    W_data = np.random.randint(-128, 127, (K, M), dtype=np.int8)
    inputs = {"X": X_data, "W": W_data}

    # warm-up
    for _ in range(repeat):
        _ = sess.run(None, inputs)

    # 多次运行并记录每次耗时
    times = []
    for _ in range(repeat):
        start = time.time()
        _ = sess.run(None, inputs)
        end = time.time()
        times.append(end - start)

    # 取中位数
    median_time = float(np.median(times))

    # 按中位数算 GFLOPS
    flops = 2 * N * M * K
    gflops = flops / median_time / 1e9

    print(f"N={N}, M={M}, K={K}, MedianTime={median_time:.6f}s, GFLOPS={gflops:.2f}")
    records.append({
        "N": N, "M": M, "K": K,
        "MedianTime_s": median_time,
        "GFLOPS": gflops
    })

# 保存 CSV
df = pd.DataFrame(records)
df.to_csv("int8_to_int32_gemm_median.csv", index=False)
print("Benchmark results saved to int8_to_int32_gemm_median.csv")

