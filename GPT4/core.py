import pandas as pd
import openai
import numpy as np
import os, tiktoken
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("GPT2_API")
import time
def askChatGPT(messages):
    MODEL = os.getenv("MODEL2")
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=1)
    return response['choices'][0]['message']['content']  # type: ignore


def start(a):
    init = [{
        "role": "user",
        "content": """我会提供一段文本，该文本由一位人物在不同场景与不同谈话中所进行的对话组成，但仅包含该人物的说辞，对话内容由\\n分隔。
你需要依据神经质、外向性、开放性、宜人性、尽责性这五个维度解释该人物的人格特点，并在最后将你的响应格式化为以 “神经质” ， “外向性”，“开放性” ，“宜人性”，“尽责性”为键，以high和low为值的 JSON 对象。你需要严格遵守下面的格式规范进行回答：
- 神经质: [对于神经质这一特征的分析内容，不要包括结果]
- 外向性: [对于外向性这一特征的分析内容，不要包括结果]
- 开放性: [对于开放性这一特征的分析内容，不要包括结果]
- 宜人性: [对于宜人性这一特征的分析内容，不要包括结果]
- 尽责性: [对于尽责性这一特征的分析内容，不要包括结果]
{"神经质": "[high或low]","外向性": "[high或low]","开放性": "[high或low]","宜人性": "[high或low]","尽责性": "[high或low]"}
其中开放性指个体对经验持开放、探求的态度。尽责性指个体在目标导向行为上的组织、坚持和动机。外向性指个体对外部世界的积极投入程度。宜人性指个体在合作与社会和谐性方面的差异。神经质指个体体验消极情绪的倾向。"""
    }, {
        "role": "assistant",
        "content": "好的，我需要你提供的文本以进行分析。"
    }]
    init.append({
        "role": "user",
        "content": a
    })
    try:
        time.sleep(30)
        return askChatGPT(init)
    except Exception as e:
        return e


def start_calc(a):
    init = [{
        "role": "user",
        "content": """我会提供一段文本，该文本由一位人物在不同场景与不同谈话中所进行的对话组成，但仅包含该人物的说辞，对话内容由\\n分隔。
你需要依据神经质、外向性、开放性、宜人性、尽责性这五个维度解释该人物的人格特点，并在最后将你的响应格式化为以 “神经质” ， “外向性”，“开放性” ，“宜人性”，“尽责性”为键，以high和low为值的 JSON 对象。你需要严格遵守下面的格式规范进行回答：
- 神经质: [对于神经质这一特征的分析内容，不要包括结果]
- 外向性: [对于外向性这一特征的分析内容，不要包括结果]
- 开放性: [对于开放性这一特征的分析内容，不要包括结果]
- 宜人性: [对于宜人性这一特征的分析内容，不要包括结果]
- 尽责性: [对于尽责性这一特征的分析内容，不要包括结果]
{"神经质": "[high或low]","外向性": "[high或low]","开放性": "[high或low]","宜人性": "[high或low]","尽责性": "[high或low]"}
其中开放性指个体对经验持开放、探求的态度。尽责性指个体在目标导向行为上的组织、坚持和动机。外向性指个体对外部世界的积极投入程度。宜人性指个体在合作与社会和谐性方面的差异。神经质指个体体验消极情绪的倾向。"""
    }, {
        "role": "assistant",
        "content": "好的，我需要你提供的文本以进行分析。"
    }]
    init.append({
        "role": "user",
        "content": a
    })
    return num_tokens_from_messages(init)


# 算价格
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        return 0