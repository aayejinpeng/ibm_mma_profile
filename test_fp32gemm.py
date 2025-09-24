import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import numpy as np
import time
import pandas as pd
import itertools

# 矩阵尺寸组合
sizes = [64, 128, 256, 512, 1024, 2048, 4096]  # 可调
records = []
repeat = 50

for N, M, K in itertools.product(sizes, sizes, sizes):
    # 输入张量信息
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, N, K])
    W = helper.make_tensor_value_info("W", TensorProto.FLOAT, [K, M])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, N, M])

    # 创建 MatMul 节点（FP32 GEMM）
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["X", "W"],
        outputs=["Y"]
    )

    # 构建图
    graph_def = helper.make_graph(
        nodes=[matmul_node],
        name="FP32GEMM_graph",
        inputs=[X, W],
        outputs=[Y],
    )

    # 构建模型
    model_def = helper.make_model(
        graph_def,
        producer_name='fp32_gemm_in_memory',
        ir_version=9,
        opset_imports=[onnx.helper.make_operatorsetid("", 19)]
    )

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 2      # 单线程执行运算
    sess_options.inter_op_num_threads = 2      # 单线程调度
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # 串行执行
    sess_options.enable_profiling = True

    # 创建 Session
    # session = ort.InferenceSession(model_path, sess_options)
    # 创建 ONNXRuntime session（内存中）
    sess = ort.InferenceSession(model_def.SerializeToString(), sess_options, providers=["CPUExecutionProvider"])
 

    # 构造随机 FP32 输入
    X_data = np.random.randn(1, N, K).astype(np.float32)
    W_data = np.random.randn(K, M).astype(np.float32)
    inputs = {"X": X_data, "W": W_data}

    # warm-up
    _ = sess.run(None, inputs)

    # 计时
    start = time.time()
    for _ in range(repeat):
        _ = sess.run(None, inputs)
    end = time.time()
    dur = end - start
    flops = 2 * N * M * K * repeat
    gflops = flops / dur / 1e9

    print(f"N={N}, M={M}, K={K}, Time={dur:.6f}s, GFLOPS={gflops:.2f}")
    records.append({
        "N": N, "M": M, "K": K, "Time_s": dur, "GFLOPS": gflops
    })

# 保存 CSV
df = pd.DataFrame(records)
df.to_csv("fp32_gemm.csv", index=False)
print("Benchmark results saved to fp32_gemm.csv")
