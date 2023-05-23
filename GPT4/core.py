import pandas as pd
import openai
import numpy as np
import os, tiktoken
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("GPT1_API")
import time
def askChatGPT(messages):
    MODEL = os.getenv("MODEL1")
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=1)
    return response['choices'][0]['message']['content']  # type: ignore


def get_arr(a):
    init = [{
        "role": "user",
        "content": """我会提供一段文本，该文本由一位人物在不同场景与不同谈话中所进行的对话组成，但仅包含该人物的说辞，对话内容由\\n分隔。
你需要依据神经质、外向性、开放性、宜人性、尽责性这五个维度分析该人物的人格特点，分析的时候你需要严格遵守下面的格式规范进行回答，且分析内容必须引用所给内容：
- [维度]: [对于该维度的分析内容]

> 其中开放性指个体对经验持开放、探求的态度。尽责性指个体在目标导向行为上的组织、坚持和动机。外向性指个体对外部世界的积极投入程度。宜人性指个体在合作与社会和谐性方面的差异。神经质指个体体验消极情绪的倾向。而且在分析完毕后，你还要给出分析的最终结果，结果只有高和低两个值，如果不知道就是不知道，以json格式保存，具体格式如下:{"神经质": "[高或低]","外向性": "[高或低]","开放性": "[高或低]","宜人性": "[高或低]","尽责性": "[高或低]"}
下面是我提供的文本: """ + a
    }]
    return init

def start(a):
    try:
        time.sleep(30)
        return askChatGPT(get_arr(a))
    except Exception as e:
        return e


def start_calc(a):
    return num_tokens_from_messages(get_arr(a), "gpt-4")


# 算价格
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens