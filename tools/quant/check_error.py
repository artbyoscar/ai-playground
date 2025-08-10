import argparse, json, numpy as np

def h2f(u16):
    x = u16.astype(np.uint32)
    s = ((x>>15)&1).astype(np.float32)
    e = ((x>>10)&31).astype(np.int32)
    m = (x & 1023).astype(np.int32)
    f = np.empty_like(s, dtype=np.float32)
    for i in range(x.size):
        if e[i]==0 and m[i]==0: f[i]=(-1.0 if s[i] else 1.0)*0.0
        else:
            E = e[i]-15
            f[i] = ((-1.0 if s[i] else 1.0) * (1.0 + m[i]/1024.0)) * (2.0**E)
    return f

def decode_col(data, scales_fp16, K, group):
    groups = (K+group-1)//group
    out = np.zeros((K,), dtype=np.float32)
    idx=0
    for g in range(groups):
        s = h2f(scales_fp16[g:g+1])[0]
        for b in range(group//2):
            if (g*group + 2*b) >= K: break
            v = int(data[idx]); idx += 1
            q0 = ((v >> 4) & 0xF) - 8
            q1 = (v & 0xF) - 8
            out[g*group + 2*b]   = s * q0
            if g*group + 2*b + 1 < K:
                out[g*group + 2*b + 1] = s * q1
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)   # KxN .npy
    ap.add_argument("--pack", required=True)      # prefix used in quantize
    ap.add_argument("--group", type=int, default=64)
    args = ap.parse_args()

    B = np.load(args.weights) # KxN
    with open(args.pack+".json") as f: desc=json.load(f)
    K,N = B.shape
    assert K==desc["K"] and N==desc["N"]

    data = np.fromfile(args.pack+".bin", dtype=np.uint8)
    Sc   = np.fromfile(args.pack+".scales.fp16.bin", dtype=np.uint16).reshape(N, -1)

    # naive decode + matmul vs FP32
    C_ref = B.T @ B   # quick self-matmul proxy: NxN
    C_q   = np.zeros_like(C_ref, dtype=np.float32)
    bytes_per_group=args.group//2
    for n in range(N):
        col = decode_col(data[n*Sc.shape[1]*bytes_per_group : (n+1)*Sc.shape[1]*bytes_per_group],
                         Sc[n], K, args.group)
        C_q[:,n] = (B.T @ col)

    diff = np.abs(C_q - C_ref)
    rel = diff.sum() / np.abs(C_ref).sum()
    print(f"max_abs={diff.max():.4e}  mean_abs={diff.mean():.4e}  rel_err={rel:.4e}")

if __name__=="__main__": main()
