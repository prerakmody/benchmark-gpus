# To run
```bash
./run_benchmark.sh
```

# Args
```bash
--volume_size <D> <H> <W>
--epochs <epochs>
--iters <iters>
--inference_iters <inference_iters>
--batch_size <batch_size>
--depth <depth>
--filters <filters>
--cleanup
```

# Sample output
```bash
========================================
       3D SEGMENTATION BENCHMARK REPORT       
========================================
Device: CPU
Volume: [64, 64, 64]
Batch Size: 2

Training Results (Avg over 2 epochs):
  Throughput : 0.43 volumes/sec
  Latency    : 3532.54 ms/batch

Inference Results:
  Throughput : 1.35 volumes/sec
  Latency    : 1486.21 ms/batch
========================================
```

```bash
========================================
       3D SEGMENTATION BENCHMARK REPORT
========================================
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
Peak Memory: 685.03 MB
Volume: [64, 64, 64]
Batch Size: 2

Training Results (Avg over 2 epochs):
  Throughput : 44.58 volumes/sec
  Latency    : 36.31 ms/batch

Inference Results:
  Throughput : 243.67 volumes/sec
  Latency    : 8.21 ms/batch
========================================
```