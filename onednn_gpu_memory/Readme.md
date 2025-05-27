# 使い方
```
icpx -fsycl onednn_gpu_memory.cpp -ldnnl -lze_loader
watch -n 1 xpu-smi ps -d 0 | grep a.out >> test.txt
```
