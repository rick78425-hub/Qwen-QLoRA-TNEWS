"""TNEWS loading and response-only SFT masking, adapted from the notebook."""
CODE_TO_NAME = {'100': '故事', '101': '文化', '102': '娱乐', '103': '体育', '104': '财经', '106': '房产', '107': '汽车', '108': '教育', '109': '科技', '110': '军事', '112': '旅游', '113': '国际', '114': '股票', '115': '农业', '116': '游戏'}
LABELS = list(CODE_TO_NAME.values())
MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'
TRAIN_PROMPT = ('你是一个中文新闻分类助手。请根据新闻标题判断所属类别。'
                '只能从以下类别中选择一个：' + '、'.join(LABELS) + '。只输出类别名称，不要解释。')

def load_tnews():
    from datasets import load_dataset
    base = 'https://huggingface.co/datasets/clue/clue/resolve/main/tnews/'
    return load_dataset('parquet', data_files={s: base + s + '-00000-of-00001.parquet' for s in ('train', 'validation')})

def label_name(label, label_feature):
    return CODE_TO_NAME[label_feature.int2str(int(label))]

def build_messages(example, label_feature):
    return [{'role': 'system', 'content': TRAIN_PROMPT},
            {'role': 'user', 'content': example['sentence']},
            {'role': 'assistant', 'content': label_name(example['label'], label_feature)}]

def token_ids(tokenizer, messages, generation_prompt):
    result = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation_prompt)
    return list(result['input_ids'] if hasattr(result, 'keys') else result)

def preprocess_sft(example, tokenizer, label_feature):
    messages = build_messages(example, label_feature)
    prompt_ids = token_ids(tokenizer, messages[:-1], True)
    full_ids = token_ids(tokenizer, messages, False)
    if full_ids[:len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids):
        raise ValueError('Chat template prefix mismatch or empty response; review tokenizer before training')
    return {'input_ids': full_ids, 'attention_mask': [1]*len(full_ids),
            'labels': [-100]*len(prompt_ids) + full_ids[len(prompt_ids):]}
