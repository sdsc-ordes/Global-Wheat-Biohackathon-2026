'''
Script to test the performance of Renku sessions during the onboarding test on Nov 16th.
'''

import multiprocessing
import os
import tempfile
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

MODEL = "GlobalWheat/GWFSS_Segformer_b4" # from HF 
DATASET = Path("examples/train-00000-of-00001.parquet")
IMAGES = [Image.open(BytesIO(row["bytes"])).convert("RGB") for row in pq.read_table(DATASET, columns=["image"]).column("image").to_pylist()]

def cgroup(name):
    '''
    Reads session's limits from a cgroup v2 file.
    '''
    path = Path("/sys/fs/cgroup") / name
    return path.read_text().strip().replace("\n", ", ") if path.exists() else "n/a"

def load(device):
    '''
    Loads the model and the sample dataset as one batch on `device`.
    '''
    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL).to(device).eval()
    images = processor(images=IMAGES, return_tensors="pt").pixel_values.to(device)  # all 10 images, (10, 3, 512, 512)
    return model, images 

def work(seconds):
    '''
    Integer addition until `seconds` pass.
    '''
    end = time.perf_counter() + seconds
    n = 0
    while time.perf_counter() < end:
        n += 1
    return n

def cpu_run():
    '''
    Runs unbound integer addition on the CPU to try and overwhelm the session.
    '''
    print("=== CPU TEST ===\n")
    print("This test will keep every CPU at 100% for 60 seconds.\nYou can stop it at any time with Ctrl+C\n")
    time.sleep(5)
    cores = os.cpu_count() or 1
    print(f"os.cpu_count(): {cores} | cgroup cpu.max: {cgroup('cpu.max')}")
    with multiprocessing.Pool(cores) as pool:
        loops = pool.map(work, [60] * cores)
    print(f"{sum(loops) / 60 / 1e6:.1f} M loops/s across {cores} processes")  
    print(f"cgroup cpu.stat: {cgroup('cpu.stat')}")  

def gpu_run():
    '''
    Runs inference on the GPU to try and go over available VRAM.
    '''
    print("=== GPU TEST ===\n")
    print("This test will run inference until it goes over available VRAM.\nYou can stop it at any time with Ctrl+C\n")
    time.sleep(5)
    if not torch.cuda.is_available():
        print("No CUDA device in this session, skipping.")
        return
    model, batch = load("cuda")
    print(f"{torch.cuda.get_device_name()}: {torch.cuda.mem_get_info()[0] / 2**30:.1f} GiB VRAM free")
    with torch.inference_mode():
        try:
            while 1:  # double the batch every pass until VRAM runs out
                start = time.perf_counter()
                model(pixel_values=batch)
                torch.cuda.synchronize()
                print(f"batch {len(batch)}: {time.perf_counter() - start:.1f}s | {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB used")
                batch = torch.cat([batch, batch])
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA out of memory at batch {len(batch)}")

def oom_run():
    '''
    Fills the session's memory to 90% of limit with copies of the test dataset, holds for 60 seconds, then frees it.
    '''
    print("=== MEMORY TEST ===\n")
    print("This test will fill 90% of the session's memory for 60 seconds, then free it.\nYou can stop it at any time with Ctrl+C\n")
    time.sleep(5)
    limit = cgroup("memory.max")
    if not limit.isdigit():
        print(f"Session memory limit unknown (memory.max: {limit}), skipping...")
        return
    limit = int(limit)
    dataset = [np.asarray(image, dtype=np.float32) for image in IMAGES]
    size = sum(array.nbytes for array in dataset)
    copies = []
    while int(cgroup("memory.current")) + size < 0.9 * limit:  # memory.current counts every process in the session
        copies.append([array.copy() for array in dataset])
        print(f"Used memory: {int(cgroup('memory.current')) / 2**30:.2f} of {limit / 2**30:.2f} GiB")
    os.system("clear")
    print(f"Memory usage: {int(cgroup('memory.current')) / limit:.0%} of the limit, now sleeping for 60 seconds")
    time.sleep(60)
    copies = None
    print(f"Freed, used memory now {int(cgroup('memory.current')) / 2**30:.2f} GiB | cgroup memory.events: {cgroup('memory.events')}")  

def huggingface_rate_limit():
    '''
    Tries to trigger rate limiting from HF.
    '''
    print("=== RATE LIMITING TEST ===\n")
    print("This test will download from HuggingFace until rate-limited.\nYou can stop it at any time with Ctrl+C\n")
    time.sleep(5)
    with tempfile.TemporaryDirectory() as cache_dir:  
        start = time.perf_counter()
        path = hf_hub_download(MODEL, "model.safetensors", cache_dir=cache_dir, token=False)
        print(f"model.safetensors: {os.path.getsize(path) / 2**20 / (time.perf_counter() - start):.1f} MiB/s")
    session = requests.Session()
    for url in (f"https://huggingface.co/{MODEL}/resolve/main/config.json", f"https://huggingface.co/api/models/{MODEL}"):
        print(f"\n--- {url} ---")
        start = time.perf_counter()
        i = 1
        while time.perf_counter() < start + 60:
            response = session.get(url, allow_redirects=False, timeout=30)
            quota = dict(field.split("=") for field in response.headers.get("RateLimit", "").split(";")[1:])  # '"api";r=255;t=152' -> {'r': '255', 't': '152'}
            print(f"#{i} at {time.perf_counter() - start:.1f}s: HTTP {response.status_code} {response.reason} | {quota.get('r', '?')} requests left for this IP, quota resets in {quota.get('t', '?')}s")
            if response.status_code == 429:
                print(f"\n--- RATE LIMITED after {i} requests, {time.perf_counter() - start:.1f}s into the test ---\n")
                break
            i += 1

def main():
    os.system("clear")
    print("=== Renku stress test script ===\n")
    print("This script will try to push to (and maybe beyond) the resource limits of the session.")
    while 1:
        mode = input("Please type in the test (cpu/gpu/oom/hf): ")
        os.system("clear")
        if mode == "cpu":
            cpu_run()
        elif mode == "gpu":
            gpu_run()
        elif mode == "oom":
            oom_run()
        elif mode == "hf":
            huggingface_rate_limit()
        else:
            print("Please choose from cpu/gpu/oom/hf.")
            continue
        run_again = input(f"Test {mode} completed. Run again? (y/n)")
        if run_again == "y":
            continue
        else:
            print("Shutting down...")
            break

if __name__ == "__main__":
    main()

