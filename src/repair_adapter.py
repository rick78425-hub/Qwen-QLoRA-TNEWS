"""Offline prefix normalization for the specific historical duplicated-prefix bug.
Writes a NEW adapter directory, then use verify_adapter to verify loading.
The original notebook repaired the live model and saved it; this helper is not a rerun of that procedure.
"""
import argparse
from pathlib import Path
from shutil import copy2

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True,type=Path); p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    from safetensors.torch import load_file,save_file
    old='base_model.model.base_model.model.'; new='base_model.model.'
    weights=load_file(str(args.source/'adapter_model.safetensors'))
    if not weights or not all(k.startswith(old) for k in weights): raise ValueError('Unexpected key pattern; no files written')
    fixed={new+k[len(old):]:v.contiguous() for k,v in weights.items()}
    if len(fixed)!=len(weights): raise ValueError('Key collision')
    args.output.mkdir(parents=True,exist_ok=False)
    save_file(fixed,str(args.output/'adapter_model.safetensors'))
    copy2(args.source/'adapter_config.json',args.output/'adapter_config.json')
    print(f'Normalized {len(fixed)} tensors. Verify reload before use.')

if __name__=='__main__': main()
