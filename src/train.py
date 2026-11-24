"""Clean-start training reference. Historical results are not a new reproduction."""
import argparse
from pathlib import Path
from .preprocess import load_tnews, preprocess_sft
from .inference import load_model

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--smoke', action='store_true', help='Only 32 train/16 validation samples and one optimizer step')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Choose a new output directory: {args.output}')
    from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Original full-training BF16 configuration requires CUDA BF16 support.')
    dataset = load_tnews()
    model, tokenizer = load_model()
    tokenizer.pad_token = tokenizer.eos_token
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32,
        target_modules=['q_proj','k_proj','v_proj','o_proj'], lora_dropout=0.05,
        bias='none', task_type='CAUSAL_LM'))
    model.print_trainable_parameters()
    labels = dataset['train'].features['label']
    train = dataset['train']; val = dataset['validation'].shuffle(seed=42).select(range(2000))
    if args.smoke:
        train = train.select(range(32)); val = val.select(range(16))
    def encode(ds):
        return ds.map(lambda row: preprocess_sft(row, tokenizer, labels), remove_columns=ds.column_names)
    train, val = encode(train), encode(val)
    kwargs = dict(output_dir=str(args.output), per_device_train_batch_size=4,
                  gradient_accumulation_steps=4, num_train_epochs=3, learning_rate=2e-4,
                  logging_steps=10, save_steps=500, eval_strategy='steps', eval_steps=500,
                  fp16=False, bf16=True, report_to='none')
    if args.smoke:
        kwargs.update(max_steps=1, logging_steps=1, eval_strategy='no', save_strategy='no')
    trainer = Trainer(model=model, args=TrainingArguments(**kwargs), train_dataset=train,
                      eval_dataset=val, data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True))
    trainer.train()
    trainer.save_model(str(args.output / 'adapter'))

if __name__ == '__main__':
    main()
