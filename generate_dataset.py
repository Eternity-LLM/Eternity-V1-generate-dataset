from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    padding_side="left"  # 生成时左填充更合理
)
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    device_map="auto",  # 自动分配到GPU（如有）
    torch_dtype=torch.bfloat8  # 降低显存占用（需GPU支持，否则用torch.float16）
    trust_remote_code=True  # 避免自定义代码加载报错
)

# 加载提示词
cn_file = open('./cn_prompts.md', encoding='utf-8')
cn_prompt = cn_file.read()
cn_file.close()
en_file = open('./en_prompts.md', encoding='utf-8')
en_prompt = en_file.read()
en_file.close()



def generate_chinese_cot(question, answer):
    # 用于为中文问题生成思维链（CoT）
    # 注意：此函数仅限于中文！！！
    messages = [
        {
            'role':'system', 
        	'content':'该助手为DeepSeek-R1，由深度求索公司制造。'
        },
        {
            'role':'user',
            'content': f'{cn_prompt}\n问题是：{question}\n答案是：{answer}'
        }
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)  # 确保输入和模型在同一设备

    outputs = model.generate(**inputs, max_new_tokens=163840)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

def generate_english_cot(question, answer):
    # 用于为英文问题生成思维链（CoT）
    # 注意：此函数仅限于英文！！！
    messages = [
        {
            'role':'system', 
        	'content':'This assistant is DeepSeek-R1, developed by DeepSeek-AI company.'
        },
        {
            'role':'user',
            'content': f'{en_prompt}\nQuestion: {question}\nAnswer: {answer}'
        }
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)  # 确保输入和模型在同一设备

    outputs = model.generate(**inputs, max_new_tokens=163840)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

