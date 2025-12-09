import argparse
import time
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import monai
from monai.networks.nets import BasicUNet, UNet
from torchsummary import summary

# --- Dataset ---
class Random3DDataset(Dataset):
    def __init__(self, epoch_len, volume_size=(128, 128, 128)):
        self.epoch_len = epoch_len
        self.volume_size = volume_size

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, idx):
        # Generate random 3D noise input (1 channel)
        img = torch.randn(1, *self.volume_size)
        
        # Generate random binary sphere mask
        D, H, W = self.volume_size
        mask = torch.zeros(self.volume_size, dtype=torch.long)
        
        # Random center and radius
        cx = np.random.randint(0, D)
        cy = np.random.randint(0, H)
        cz = np.random.randint(0, W)
        max_r = min(D, H, W) // 4
        r = np.random.randint(5, max(6, max_r))
        
        # Create grid
        z, y, x = np.ogrid[:D, :H, :W]
        dist_sq = (z - cx)**2 + (y - cy)**2 + (x - cz)**2
        mask[dist_sq <= r**2] = 1
        
        return img, mask

# --- Utils ---
def get_gpu_utilization():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            encoding='utf-8'
        )
        # Assuming single GPU or taking the first one
        util, mem = result.strip().split('\n')[0].split(',')
        return float(util), float(mem)
    except Exception:
        return 0.0, 0.0

def print_params(args):
    print("VOLUME_D=" + str(args.volume_size[0]))
    print("VOLUME_H=" + str(args.volume_size[1]))
    print("VOLUME_W=" + str(args.volume_size[2]))
    print("EPOCHS=" + str(args.epochs))
    print("ITERS=" + str(args.iters))
    print("INFERENCE_ITERS=" + str(args.inference_iters))
    print("BATCH_SIZE=" + str(args.batch_size))
    print("DEPTH=" + str(args.depth))
    print("FILTERS=" + str(args.filters))
    print("CLEANUP=" + str(args.cleanup).lower())


# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="3D Segmentation GPU Benchmark")
    parser.add_argument("--volume_size", type=int, nargs=3, default=[128, 128, 128], help="Input volume size D H W")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--iters", type=int, default=50, help="Iterations per epoch")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--depth", type=int, default=3, help="UNet depth")
    parser.add_argument("--filters", type=int, default=32, help="Base filters for UNet")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    parser.add_argument("--inference_iters", type=int, default=20, help="Number of inference iterations")
    
    parser.add_argument("--cleanup", action="store_true", help="Cleanup flag for printing")
    
    args = parser.parse_args()
    
    print_params(args)
    print(f"--- Configuration ---")
    print(f"Device: {args.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
    print(f"Volume Size: {args.volume_size}")
    print(f"Model: {args.depth} depth, {args.filters} filters (MONAI UNet)")
    print(f"Training: {args.epochs} epochs, {args.iters} iters/epoch, batch {args.batch_size}")
    print(f"Inference: {args.inference_iters} iters, batch {args.batch_size}")
    print(f"---------------------")

    # Data
    dataset = Random3DDataset(args.iters * args.batch_size, tuple(args.volume_size))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    
    # Construct channels and strides for UNet based on depth and filters
    # depth=3 means 3 downsampling layers.
    # We need len(channels) = depth + 1 (bottom layer) + ? BasicUNet had 6 layers fixed.
    # Standard UNet:
    # channels needs to encompass the encoder path.
    # strides needs to be len(channels) - 1.
    
    # Let's say depth=3. We want 3 downsamples.
    # channels: [base, base*2, base*4, base*8] -> 4 levels.
    # strides: [2, 2, 2] -> 3 downsamples.
    
    channels = [args.filters * (2**i) for i in range(args.depth + 1)]
    strides = [2] * args.depth
    
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=tuple(channels),
        strides=tuple(strides)
    )
    model.to(args.device)

    # --- Model Summary and Param Count ---
    try:
        from torchsummary import summary
        print("\n--- Model Summary ---")
        # torchsummary requires input size (C, D, H, W) without batch dimension
        # The input volume size is args.volume_size (D, H, W)
        summary(model, (1, *args.volume_size))
    except ImportError:
        print("torchsummary not installed, skipping detailed summary.")
    except Exception as e:
        print(f"Failed to run torchsummary: {e}")

    total_params = sum(p.numel() for p in model.parameters())
    param_status = "> 1M" if total_params > 1000000 else "< 1M"
    print(f"Total Parameters: {total_params:,} ({param_status})") 
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # --- Training Loop ---
    training_metrics = []
    if args.epochs > 0:
        print("\n--- Starting Training Benchmark ---")
        for epoch in range(args.epochs):
            model.train()
            epoch_start = time.time()
            batch_times = []
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
            for i, (inputs, targets) in enumerate(pbar):
                t0 = time.time()
                
                inputs = inputs.to(args.device)
                targets = targets.to(args.device)
                
                optimizer.zero_grad()
                
                if args.device == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    torch.cuda.synchronize()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                t1 = time.time()
                batch_times.append(t1 - t0)
                
                if i % 10 == 0:
                    util, mem = get_gpu_utilization()
                    pbar.set_postfix({'loss': f"{loss.item():.4f}", 'gpu_util': f"{util}%", 'mem': f"{mem}MB"})
            
            epoch_dur = time.time() - epoch_start
            avg_batch = np.mean(batch_times)
            throughput = (args.iters * args.batch_size) / epoch_dur
            
            print(f"Epoch {epoch+1}: {throughput:.2f} vol/s | Avg Batch: {avg_batch*1000:.2f}ms")
            training_metrics.append({
                "epoch": epoch + 1,
                "throughput": throughput,
                "avg_batch_ms": avg_batch * 1000
            })

    # --- Inference Loop ---
    inference_metrics = {}
    if args.inference_iters > 0:
        print("\n--- Starting Inference Benchmark ---")
        model.eval()
        # Create a dummy input for inference to avoid dataloader overhead in pure model test
        dummy_input = torch.randn(args.batch_size, 1, *args.volume_size).to(args.device)
        
        # Warmup
        print("Warmup...")
        with torch.no_grad():
            for _ in range(5):
                if args.device == 'cuda':
                    with torch.cuda.amp.autocast():
                        _ = model(dummy_input)
                    torch.cuda.synchronize()
                else:
                    _ = model(dummy_input)
        
        print(f"Running {args.inference_iters} inference steps...")
        inf_times = []
        with torch.no_grad():
            for _ in tqdm(range(args.inference_iters)):
                t0 = time.time()
                if args.device == 'cuda':
                    with torch.cuda.amp.autocast():
                        _ = model(dummy_input)
                    torch.cuda.synchronize()
                else:
                    _ = model(dummy_input)
                t1 = time.time()
                inf_times.append(t1 - t0)
        
        avg_inf = np.mean(inf_times)
        inf_throughput = args.batch_size / avg_inf
        inference_metrics = {
            "avg_latency_ms": avg_inf * 1000,
            "throughput": inf_throughput
        }

    # --- Final Report ---
    print("\n" + "="*40)
    print("       3D SEGMENTATION BENCHMARK REPORT       ")
    print(f"       (Parameters: {total_params/1e6:.2f}M)")
    print("="*40)
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Peak Memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    else:
        print("Device: CPU")
        
    print(f"Volume: {args.volume_size}")
    print(f"Batch Size: {args.batch_size}")
    
    if training_metrics:
        avg_train_fps = np.mean([m['throughput'] for m in training_metrics])
        avg_train_lat = np.mean([m['avg_batch_ms'] for m in training_metrics])
        print(f"\nTraining Results (Avg over {args.epochs} epochs):")
        print(f"  Throughput : {avg_train_fps:.2f} volumes/sec")
        print(f"  Latency    : {avg_train_lat:.2f} ms/batch")
        
    if inference_metrics:
        print(f"\nInference Results:")
        print(f"  Throughput : {inference_metrics['throughput']:.2f} volumes/sec")
        print(f"  Latency    : {inference_metrics['avg_latency_ms']:.2f} ms/batch")
    
    print("="*40)
    print_params(args)

if __name__ == "__main__":
    main()
