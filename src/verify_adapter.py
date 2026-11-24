"""Compare every saved adapter tensor with the reloaded PEFT state dict."""
import argparse
from pathlib import Path
from .inference import load_model

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter', required=True, type=Path)
    args=p.parse_args()
    import torch
    from safetensors.torch import load_file
    from peft import get_peft_model_state_dict
    saved=load_file(str(args.adapter/'adapter_model.safetensors'))
    model,_=load_model(args.adapter)
    loaded=get_peft_model_state_dict(model)
    if set(saved)!=set(loaded):
        raise RuntimeError(f'Key mismatch: missing={sorted(set(saved)-set(loaded))[:5]}, extra={sorted(set(loaded)-set(saved))[:5]}')
    differences=[]
    for key,weight in saved.items():
        actual=loaded[key].detach().cpu().float()
        if actual.shape!=weight.shape: raise RuntimeError(f'Shape mismatch: {key}')
        if not torch.isfinite(actual).all() or not torch.isfinite(weight).all(): raise RuntimeError(f'Nonfinite: {key}')
        differences.append((actual-weight.float()).abs().max().item())
    b_weights=[w for k,w in saved.items() if '.lora_B.' in k]
    if not b_weights or not any(w.abs().max().item()>0 for w in b_weights): raise RuntimeError('No nonzero LoRA B weights')
    print(f'Matched tensors: {len(saved)}; max absolute difference: {max(differences):.9g}')
    if max(differences)!=0: raise RuntimeError('Loaded weights differ; review dtype and mapping before evaluation')
    print('Adapter verification passed')

if __name__=='__main__': main()
