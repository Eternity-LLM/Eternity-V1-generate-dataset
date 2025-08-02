from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


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

unstructured_cn_file = open('./unstructured_cn.md', encoding='utf-8')
unstructured_cn = unstructured_cn_file.read()
unstructured_cn_file.close()

unstructured_en_file = open('./unstructured_en.md', encoding='utf-8')
unstructured_en = unstructured_en_file.read()
unstructured_en_file.close()


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
    cot = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    cot = cot[cot.find('</think>')+8 :]
    return cot

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
    cot = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    cot = cot[cot.find('</think>')+8 :]
    return cot

def generate_answer_and_cot(question):
    # 用于为中文问题生成答案和思维链（CoT）
    # 注意：此函数仅限于数据中只有问题没有答案的数据。
    # 如果数据中有答案，请使用generate_chinese_cot或generate_english_cot函数。
    # 此函数返回2个字符串：思维链和答案。
    messages = [
        {
            'role':'system', 
        	'content':'该助手为DeepSeek-R1，由深度求索公司制造。'
        },
        {
            'role':'user',
            'content': question
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
    output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    cot = output[output.find('<think>')+7 : output.find('</think>')]
    answer = output[output.find('</think>')+8 :]
    return cot, answer

def generate_data_for_unstructured_cn(text):
    # 用于为未结构化文本生成问题、思维链（CoT）和答案。
    # 注意：此函数仅限于未结构化文本中文数据。
    messages = [
        {
            'role':'system', 
        	'content':'该助手为DeepSeek-R1，由深度求索公司制造。'
        },
        {
            'role':'user',
            'content': f'{unstructured_cn}{text}'
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
    output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    question = output[output.find('</think>')+8:]
    cot, answer = generate_answer_and_cot(question)
    return question, cot, answer

def generate_data_for_unstructured_en(text):
    # 用于为未结构化文本生成问题、思维链（CoT）和答案。
    # 注意：此函数仅限于未结构化文本英文数据。
    messages = [
        {
            'role':'system', 
        	'content':'This assistant is DeepSeek-R1, developed by DeepSeek-AI company.'
        },
        {
            'role':'user',
            'content': f'{unstructured_en}{text}'
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
    output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    question = output[output.find('</think>')+8:]
    cot, answer = generate_answer_and_cot(question)
    return question, cot, answer

def save_data(question, cot, answer, file_name):
    for text in [question, cot, answer]:
        temp = ''
        for i in range(len(question)):
            if text[i] == '\n':
                temp += '\\n'
            elif text[i] == '\\':
                temp += '\\\\'
            else:
                temp += text[i]
        text = temp
    with open(f'./{file_name}', mode='w', encoding='utf-8') as f:
        f.write('{\'question\':%s, \'cot\':%s, \'answer\':%s}\n'%(question, cot, answer))
    return


