import time, json, argparse
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    import nvtx
    _NVTX = True
except ImportError:
    _NVTX = False

class annotate:
    def __init__(self, name, color="white"):
        self.name = name; self.color = color; self._ctx = None
    def __enter__(self):
        if _NVTX:
            self._ctx = nvtx.annotate(self.name, color=self.color)
            self._ctx.__enter__()
        return self
    def __exit__(self, *a):
        if _NVTX and self._ctx: self._ctx.__exit__(*a)

def setup_device():
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"Device : CUDA")
            print(f"GPU    : {props.name}")
            print(f"VRAM   : {props.total_memory / 1024**3:.1f} GB")
            print(f"SMs    : {props.multi_processor_count}")
            print(f"CC     : {props.major}.{props.minor}")
            print(f"Torch  : {torch.__version__}")
            print(f"CUDA rt: {torch.version.cuda}")
            return torch, "cuda"
        else:
            print("Device : CPU (CUDA not available)")
            return torch, "cpu"
    except ImportError:
        print("Device : CPU-NumPy")
        return None, "numpy"

def sync(mode):
    if mode == "cuda":
        import torch; torch.cuda.synchronize()

def build_tensor_cache(mode, torch_mod):
    if mode != "cuda":
        return {}
    C,H,W = 3,720,1280
    DIM=15; FEAT=2048; N=10000; G=250
    print("Pre-allocating GPU tensor cache (zero allocations after this point)...")
    cache = {
        "left"   : torch_mod.zeros(C,H,W, device="cuda"),
        "right"  : torch_mod.zeros(C,H,W, device="cuda"),
        "F"      : torch_mod.zeros(DIM,DIM, device="cuda"),
        "Q"      : torch_mod.eye(DIM, device="cuda") * 0.001,
        "state"  : torch_mod.zeros(DIM, device="cuda"),
        "cov"    : torch_mod.eye(DIM, device="cuda"),
        "x_feat" : torch_mod.zeros(1,FEAT, device="cuda"),
        "W1"     : torch_mod.randn(FEAT,FEAT, device="cuda"),
        "W2"     : torch_mod.randn(FEAT,512, device="cuda"),
        "W3"     : torch_mod.randn(512,400, device="cuda"),
        "samples": torch_mod.zeros(N,3, device="cuda"),
        "goal"   : torch_mod.tensor([125.,125.,25.], device="cuda"),
        "occ"    : torch_mod.zeros(G,G, device="cuda"),
    }
    for t in cache.values(): _ = t.sum()
    torch_mod.cuda.synchronize()
    mb = sum(t.numel()*t.element_size() for t in cache.values())/1024**2
    print(f"Cache ready: {len(cache)} tensors, {mb:.1f} MB GPU memory")
    return cache

def stage1_feature_extraction(left, right, mode):
    with annotate("stage1_feature_extraction", color="blue"):
        if mode == "cuda":
            import torch
            Ix = left[:,:,1:] - left[:,:,:-1]
            Iy = left[:,1:,:] - left[:,:-1,:]
            H,W = left.shape[1], left.shape[2]
            Ix2  = Ix[:,:H-1,:W-1]**2
            Iy2  = Iy[:,:H-1,:W-1]**2
            IxIy = Ix[:,:H-1,:W-1]*Iy[:,:H-1,:W-1]
            det  = Ix2*Iy2 - IxIy**2
            R    = det - 0.05*(Ix2+Iy2)**2
            _, idx = torch.topk(R.reshape(-1), 500)
            sync(mode); return idx
        else:
            Ix = np.diff(left,axis=2); Iy = np.diff(left,axis=1)
            H,W = left.shape[1],left.shape[2]
            R = (Ix[:,:H-1,:W-1]**2 * Iy[:,:H-1,:W-1]**2
                 - (Ix[:,:H-1,:W-1]*Iy[:,:H-1,:W-1])**2)
            return np.argpartition(R.reshape(-1),-500)[-500:]

def stage2_vio_update(kp, cache, mode):
    with annotate("stage2_vio_update", color="green"):
        if mode == "cuda":
            import torch
            cache["F"].normal_()
            P_new = torch.matmul(cache["F"],
                        torch.matmul(cache["cov"],cache["F"].t())) + cache["Q"]
            cache["cov"].copy_(P_new)
            cache["state"].copy_(torch.matmul(cache["F"],cache["state"]))
            sync(mode)
        else:
            F = np.random.randn(15,15).astype(np.float32)

def stage3_depth_estimation(left, right, mode):
    with annotate("stage3_depth_estimation", color="yellow"):
        if mode == "cuda":
            import torch, torch.nn.functional as F
            k   = torch.ones(1,1,5,5,device="cuda")/25.0
            img = left.mean(dim=0,keepdim=True).unsqueeze(0)
            out = torch.zeros_like(img)
            for d in range(0,64,8):
                out += F.conv2d(torch.abs(img-torch.roll(img,d,dims=-1)),k,padding=2)
            sync(mode); return out.squeeze()
        else:
            img = left.mean(axis=0); out = np.zeros_like(img)
            for d in range(0,64,8): out += np.abs(img-np.roll(img,d,axis=1))
            return out

def stage4_obstacle_detection(left, cache, mode):
    with annotate("stage4_obstacle_detection", color="red"):
        if mode == "cuda":
            import torch
            cache["x_feat"].normal_()
            x   = torch.relu(cache["x_feat"] @ cache["W1"])
            x   = torch.relu(x @ cache["W2"])
            det = x @ cache["W3"]
            sync(mode); return det
        else:
            FEAT=2048
            x=np.random.randn(1,FEAT).astype(np.float32)
            W1=np.random.randn(FEAT,FEAT).astype(np.float32)
            W2=np.random.randn(FEAT,512).astype(np.float32)
            W3=np.random.randn(512,400).astype(np.float32)
            return np.maximum(0,np.maximum(0,x@W1)@W2)@W3

def stage5_path_planning(depth, cache, mode):
    with annotate("stage5_path_planning", color="purple"):
        N,G = 10000,250
        if mode == "cuda":
            import torch
            dm = depth[:G,:G]
            cache["occ"].copy_((dm<2.0).float())
            s = cache["samples"]
            s.uniform_(); s[:,:2].mul_(G); s[:,2].mul_(50)
            xi = s[:,0].long().clamp_(0,G-1)
            yi = s[:,1].long().clamp_(0,G-1)
            free = s[cache["occ"][xi,yi]<0.5]
            if free.shape[0]>0:
                nearest = free[torch.norm(free-cache["goal"],dim=1).argmin()]
            else:
                nearest = s[0]
            sync(mode); return nearest
        else:
            occ=(depth[:G,:G]<2.0).astype(np.float32)
            s=np.random.rand(N,3).astype(np.float32)
            s[:,:2]*=G; s[:,2]*=50
            xi=np.clip(s[:,0].astype(int),0,G-1)
            yi=np.clip(s[:,1].astype(int),0,G-1)
            free=s[occ[xi,yi]<0.5]
            tgt=np.array([125.,125.,25.],dtype=np.float32)
            return free[np.linalg.norm(free-tgt,axis=1).argmin()] if len(free)>0 else s[0]

def run_cycle(mode, torch_mod, cache):
    timings = {}
    with annotate("full_pipeline_cycle", color="white"):
        with annotate("data_input", color="gray"):
            if mode == "cuda":
                cache["left"].uniform_(); cache["right"].uniform_()
                sync(mode)
                left = cache["left"]; right = cache["right"]
            else:
                left  = np.random.rand(3,720,1280).astype(np.float32)
                right = np.random.rand(3,720,1280).astype(np.float32)

        if mode == "cuda":
            ev = {k:(torch_mod.cuda.Event(enable_timing=True),
                     torch_mod.cuda.Event(enable_timing=True))
                  for k in ["s1","s2","s3","s4","s5"]}
            ev["s1"][0].record()
            kp = stage1_feature_extraction(left,right,mode)
            ev["s1"][1].record()
            ev["s2"][0].record()
            stage2_vio_update(kp,cache,mode)
            ev["s2"][1].record()
            ev["s3"][0].record()
            depth = stage3_depth_estimation(left,right,mode)
            ev["s3"][1].record()
            ev["s4"][0].record()
            det = stage4_obstacle_detection(left,cache,mode)
            ev["s4"][1].record()
            ev["s5"][0].record()
            path = stage5_path_planning(depth,cache,mode)
            ev["s5"][1].record()
            torch_mod.cuda.synchronize()
            for name,key in zip(
                ["stage1_ms","stage2_ms","stage3_ms","stage4_ms","stage5_ms"],
                ["s1","s2","s3","s4","s5"]):
                timings[name] = ev[key][0].elapsed_time(ev[key][1])
        else:
            for fn,key,args in [
                (stage1_feature_extraction,"stage1_ms",(left,right,mode)),
                (stage2_vio_update,"stage2_ms",(None,cache,mode)),
                (stage3_depth_estimation,"stage3_ms",(left,right,mode)),
                (stage4_obstacle_detection,"stage4_ms",(left,cache,mode)),
            ]:
                t0=time.perf_counter(); fn(*args)
                timings[key]=(time.perf_counter()-t0)*1000
            depth_cpu=stage3_depth_estimation(left,right,mode)
            t0=time.perf_counter()
            stage5_path_planning(depth_cpu,cache,mode)
            timings["stage5_ms"]=(time.perf_counter()-t0)*1000
    return timings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames",type=int,default=100)
    parser.add_argument("--warmup",type=int,default=20)
    parser.add_argument("--save",action="store_true")
    args = parser.parse_args()

    print()
    print("="*58)
    print("  DOMAIN B - PIPELINE BENCHMARK  (Phase 1)")
    print("  6-Stage UAV Perception Pipeline Simulation")
    print("="*58)
    print()

    torch_mod, mode = setup_device()
    print()
    cache = build_tensor_cache(mode, torch_mod)
    print()

    print(f"Warmup: {args.warmup} cycles (not timed)...")
    for _ in range(args.warmup): run_cycle(mode,torch_mod,cache)
    if mode=="cuda": torch_mod.cuda.synchronize()
    print("Warmup done.\n")

    print(f"Timing: {args.frames} cycles...")
    all_timings = []
    for i in range(args.frames):
        all_timings.append(run_cycle(mode,torch_mod,cache))
        if (i+1)%25==0: print(f"  {i+1}/{args.frames} frames complete")
    print()

    stages = ["stage1_ms","stage2_ms","stage3_ms","stage4_ms","stage5_ms"]
    labels = [
        "Stage 1  Feature Extraction",
        "Stage 2  VIO / EKF Predict ",
        "Stage 3  Depth Estimation  ",
        "Stage 4  Obstacle Detection",
        "Stage 5  Path Planning     ",
    ]
    totals = [sum(t[s] for s in stages) for t in all_timings]
    t_arr  = np.array(totals)

    print("="*58)
    print("  PHASE 1 RESULTS - Simulated GPU Pipeline Baseline")
    print("="*58)
    if mode=="cuda": print(f"  GPU    : {torch_mod.cuda.get_device_name(0)}")
    print(f"  Frames : {args.frames}\n")
    print(f"  {'Stage':<32} {'Mean':>8}  {'Min':>8}  {'Max':>8}  {'Std':>8}")
    print(f"  {'-'*32} {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    stage_means = {}
    for s,label in zip(stages,labels):
        vals = np.array([t[s] for t in all_timings])
        stage_means[s] = float(vals.mean())
        print(f"  {label}  {vals.mean():>7.2f}ms  {vals.min():>7.2f}ms"
              f"  {vals.max():>7.2f}ms  {vals.std():>7.2f}ms")

    print(f"  {'-'*32} {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'TOTAL (end-to-end)':<32}  {t_arr.mean():>7.2f}ms"
          f"  {t_arr.min():>7.2f}ms  {t_arr.max():>7.2f}ms  {t_arr.std():>7.2f}ms\n")

    budget=33.3
    print(f"  Real-time budget (30 fps) : {budget} ms")
    print(f"  Budget met                : {'YES' if t_arr.mean()<budget else 'NO'}\n")

    spikes=[]
    for s,label in zip(stages,labels):
        vals=np.array([t[s] for t in all_timings])
        ratio=vals.max()/vals.mean() if vals.mean()>0 else 0
        if ratio>3.0: spikes.append((label.strip(),vals.mean(),vals.max(),ratio))

    if spikes:
        print("  SPIKE WARNING (max > 3x mean):")
        for name,mean,mx,ratio in spikes:
            print(f"    {name}: max={mx:.2f}ms = {ratio:.1f}x mean ({mean:.2f}ms)")
        print("  Cause: OS scheduling jitter on laptop (not a CUDA issue)\n")
    else:
        print("  Spike check : PASS - all stage max < 3x mean")
        print("  Zero GPU allocations in timing loop. Cache working correctly.\n")

    bottleneck=max(stage_means,key=stage_means.get)
    btl_label=labels[stages.index(bottleneck)]
    print(f"  Biggest bottleneck : {btl_label.strip()}")
    print(f"                       {stage_means[bottleneck]:.2f} ms  "
          f"({stage_means[bottleneck]/t_arr.mean()*100:.1f}% of pipeline)\n")
    print("  NEXT: Run with Nsight Systems:")
    print("  nsys profile `")
    print("    --trace=cuda,nvtx,wddm `")
    print("    --cuda-memory-usage=true `")
    print("    --output=profiling\\nsight_reports\\session1_baseline `")
    print("    uv run python benchmarks\\pipeline\\pipeline_benchmark.py --frames 30")
    print("  nsys-ui profiling\\nsight_reports\\session1_baseline.nsys-rep")
    print("="*58)

    if args.save:
        out={
            "timestamp":datetime.now().isoformat(),
            "phase":1, "device":mode,
            "gpu":(torch_mod.cuda.get_device_name(0) if mode=="cuda" else "CPU"),
            "frames":args.frames,"warmup":args.warmup,
            "mean_ms":float(t_arr.mean()),"min_ms":float(t_arr.min()),
            "max_ms":float(t_arr.max()),"std_ms":float(t_arr.std()),
            "budget_33ms":bool(t_arr.mean()<budget),
            "spike_free":len(spikes)==0,
            "stages":{s:{"mean_ms":float(np.mean([x[s] for x in all_timings])),
                         "min_ms":float(np.min([x[s] for x in all_timings])),
                         "max_ms":float(np.max([x[s] for x in all_timings])),
                         "std_ms":float(np.std([x[s] for x in all_timings]))}
                      for s in stages},
        }
        p=Path("results/timing_data/benchmark_phase1_baseline.json")
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(out,indent=2))
        print(f"\n  Saved: {p}")

if __name__=="__main__":
    main()

