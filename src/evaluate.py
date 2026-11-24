"""Evaluate one model at a time, to avoid keeping two models in GPU memory."""
import argparse
import json
from pathlib import Path
from .preprocess import load_tnews, label_name
from .inference import load_model, generate_category

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter', type=Path, default=None, help='Omit for base model')
    p.add_argument('--limit', type=int, default=None, help='Optional prefix for smoke testing')
    p.add_argument('--output', type=Path, default=None, help='Optional new JSON report path')
    args = p.parse_args()
    if args.limit is not None and args.limit < 1: p.error('--limit must be positive')
    if args.output and args.output.exists(): p.error('Output already exists; choose a new filename')
    from sklearn.metrics import accuracy_score, classification_report
    from tqdm import tqdm
    ds = load_tnews(); labels = ds['train'].features['label']; val = ds['validation']
    if args.limit is not None: val = val.select(range(min(args.limit,len(val))))
    model, tokenizer = load_model(args.adapter)
    y_true, y_pred = [], []
    for row in tqdm(val):
        y_true.append(label_name(row['label'], labels))
        y_pred.append(generate_category(model, tokenizer, row['sentence']))
    report = {'model': 'adapter' if args.adapter else 'base', 'samples': len(val),
              'accuracy': accuracy_score(y_true,y_pred),
              'classification_report': classification_report(y_true,y_pred,zero_division=0,output_dict=True)}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__ == '__main__': main()
