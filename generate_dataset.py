from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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

messages = [{"role": "user", "content": "Who are you?"}]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)  # 确保输入和模型在同一设备

# 调整max_new_tokens为合适长度
outputs = model.generate(**inputs, max_new_tokens=512)
# 解码时只取新增部分（排除输入）
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
