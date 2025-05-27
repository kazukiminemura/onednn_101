# 使い方
```
icpx -fsycl onednn_gpu_memory.cpp -ldnnl -lze_loader
./a.out & timeout 600 watch -n 1 "xpu-smi ps -d 0 | grep a.out >> test.txt"

icpx -fsycl onednn_gpu_memory_f32u4f32.cpp -ldnnl -lze_loader -o onednn_gpu_memory_f32u4f32
./onednn_gpu_memory_f32u4f32 & timeout 600 watch -n 1 "xpu-smi ps -d 0 | grep onednn_gpu_mem >> test.txt"
```
