Test onnxruntime In IBM Power10 S1022

https://cloud.ibm.com/power/overview

1 Dedicated Processors

test in smt=2

cpuinfo in 

https://github.com/aayejinpeng/ibm_mma_profile/blob/master/lscpu.dump

https://github.com/aayejinpeng/ibm_mma_profile/blob/master/ppc64_cpu.dump

onnx use mma by 

https://community.ibm.com/community/user/blogs/daniel-schenker/2024/02/28/how-to-run-batch-inferencing-with-onnxruntime-on-i

bert_q8 = https://github.com/onnx/models/blob/main/validated/text/machine_comprehension/bert-squad/model/bertsquad-12-int8.onnx
llama_q8 = https://huggingface.co/onnx-community/Llama-3.2-1B-Instruct/blob/main/onnx/model_int8.onnx
resnet50_q8 = https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet50-v1-12-int8.onnx