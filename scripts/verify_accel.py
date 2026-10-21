# scripts/verify_accel.py
import os, sys, torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

def pick_device():
    try:
        import torch_npu
        if torch.npu.is_available():
            return "npu:0"
    except ImportError:
        pass
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"

device = pick_device()
print(f"device: {device}")
print(f"torch: {torch.__version__}, cuda: {torch.version.cuda}")

if device == "cpu":
    print("WARNING: running on CPU, accel not detected", file=sys.stderr)
    sys.exit(2)

first = next(
    l.split("#", 1)[0].strip()
    for l in Path("docker/models.txt").read_text().splitlines()
    if l.split("#", 1)[0].strip()
)
st = SentenceTransformer(first, device="cpu")

if device.startswith("npu"):
    for m in st.modules():
        if hasattr(m, "auto_model"):
            m.auto_model.to(device).eval()
            break
    tok = st.tokenizer
    enc = tok(["offline smoke test"], padding=True, truncation=True,
              max_length=512, return_tensors="pt")
    with torch.no_grad():
        out = m.auto_model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        )
    emb = out[0]
    mask = enc["attention_mask"].unsqueeze(-1).expand(emb.size()).float().to(device)
    pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    vec = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()
else:
    st.to(device)
    vec = st.encode(["offline smoke test"], normalize_embeddings=True)

print(f"accel smoke ok on {device}: dim={vec.shape[-1]}")