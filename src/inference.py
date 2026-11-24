"""Load 4-bit Qwen and generate categories using the historical evaluation prompt."""
from .preprocess import MODEL_ID, LABELS
EVAL_PROMPT = '\n            你是一个中文新闻分类助手。\n            请根据新闻标题判断所属类别。\n            只能从以下类别中选择一个：\n            故事、文化、娱乐、体育、财经、房产、汽车、教育、科技、军事、旅游、国际、股票、农业、游戏。\n            只输出类别名称，不要解释。\n            '

def load_model(adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError('This project uses the original CUDA/bitsandbytes 4-bit setup.')
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                            bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
    return model.eval(), tokenizer

def normalize_prediction(text):
    return next((label for label in LABELS if label in text), '其他')

def generate_category(model, tokenizer, text):
    import torch
    messages = [{'role':'system','content':EVAL_PROMPT}, {'role':'user','content':text}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    prediction = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return normalize_prediction(prediction.strip())
