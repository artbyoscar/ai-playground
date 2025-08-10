import argparse, json, numpy as np, struct, os

def pack_col_int4(col, group=64):
    K = col.shape[0]
    out_data = bytearray()
    scales = []
    eps = 1e-8
    for g in range(0, K, group):
        seg = col[g:g+group]
        maxabs = float(np.max(np.abs(seg))) if seg.size else 0.0
        s = max(maxabs/7.0, eps)
        scales.append(s)
        q = np.round(seg / s).astype(np.int32)
        q = np.clip(q, -8, 7).astype(np.int8)
        # pad to even count
        if q.size % 2 == 1:
            q = np.concatenate([q, np.zeros(1, dtype=np.int8)])
        for i in range(0, q.size, 2):
            b = ((int(q[i]) + 8) & 0xF) << 4 | ((int(q[i+1]) + 8) & 0xF)
            out_data.append(b)
    return out_data, np.array(scales, dtype=np.float16)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="B as .npy with shape (K,N)")
    ap.add_argument("--out",      required=True, help="output prefix")
    ap.add_argument("--group", type=int, default=64)
    args = ap.parse_args()

    B = np.load(args.weights)  # K x N
    assert B.ndim==2
    K,N = B.shape

    data = bytearray()
    all_scales = []

    for n in range(N):
        col = B[:,n].astype(np.float32, copy=False)
        d, s = pack_col_int4(col, group=args.group)
        data.extend(d)
        all_scales.append(s)

    scales = np.stack(all_scales, axis=0) # N x groups
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out+".bin", "wb") as f: f.write(data)
    scales.view(np.uint16).tofile(args.out+".scales.fp16.bin")

    desc = {
        "layout":"colwise_k_blocks",
        "group_size": args.group,
        "scale": "fp16",
        "K": int(K), "N": int(N),
        "bytes_per_group": args.group//2,
    }
    with open(args.out+".json","w") as f: json.dump(desc, f, indent=2)
    print("wrote:", args.out+".{bin,scales.fp16.bin,json}")

if __name__=="__main__": main()
