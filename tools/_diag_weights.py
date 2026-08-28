import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from gguf.constants import GGMLQuantizationType
from safetensors import safe_open

def logical_shape(t):
    return tuple(int(d) for d in reversed(tuple(t.shape)))

reader = GGUFReader(r"models/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q5_K_M.gguf")
tmap = {t.name: t for t in reader.tensors}

emb = tmap["token_embd.weight"]
print("token_embd gguf type:", int(emb.tensor_type), "logical:", logical_shape(emb), "data shape:", emb.data.shape, "dtype:", emb.data.dtype)

# dequantize full embedding, compare row of token 104582
emb_f32 = dequantize(emb.data, emb.tensor_type)
print("emb deq shape:", emb_f32.shape)
row_gg = emb_f32[104582]

st = safe_open(r"models/Qwen3.5-4B-int4-AutoRound/model-00001-of-00005.safetensors", framework="pt")
names = list(st.keys())
emb_name = [n for n in names if "embed" in n]
print("st embed tensor:", emb_name[:5])

# find which shard contains embed_tokens - check all shards
import glob
emb_row_st = None
for f in glob.glob(r"models/Qwen3.5-4B-int4-AutoRound/model-0000*-of-00005.safetensors"):
    with safe_open(f, framework="pt") as sf:
        if "model.language_model.embed_tokens.weight" in sf.keys():
            emb_row_st = sf.get_tensor("model.language_model.embed_tokens.weight")[104582].float()
            print("found embed in", f)
            break

print("gg emb row[:8]:", row_gg[:8])
print("st emb row[:8]:", emb_row_st[:8])
cos = torch.nn.functional.cosine_similarity(torch.tensor(row_gg, dtype=torch.float32), emb_row_st, dim=0)
print("embed row cosine:", float(cos))

# layer 0 attn norm
norm_gg = dequantize(tmap["blk.0.attn_norm.weight"].data, tmap["blk.0.attn_norm.weight"].tensor_type)
with safe_open(r"models/Qwen3.5-4B-int4-AutoRound/model-00001-of-00005.safetensors", framework="pt") as sf:
    k0 = [k for k in sf.keys()][:20]
    print("st shard1 keys sample:", k0)

# compare a dequantized linear: blk.0 is linear_attention (attn_qkv)
qkv = tmap["blk.0.attn_qkv.weight"]
qkv_f32 = dequantize(qkv.data, qkv.tensor_type)
print("qkv deq shape:", qkv_f32.shape, "row0[:6]:", qkv_f32[0][:6])

# metadata config dump
arch = str(reader.fields["general.architecture"].contents())
print("arch:", arch)
for key in sorted(reader.fields.keys()):
    if key.startswith(arch) and key not in (f"{arch}.rope.dimension_sections",):
        try:
            v = reader.fields[key].contents()
            print(key, "=", v)
        except Exception as e:
            print(key, "ERR", e)
