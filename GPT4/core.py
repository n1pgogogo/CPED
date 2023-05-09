import pandas as pd
import openai
import numpy as np
from dotenv import load_dotenv
import os
import tiktoken

load_dotenv()

api_key = os.getenv("GPT_API")
openai.api_key = api_key


def askChatGPT(messages):
    MODEL = "gpt-4"
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=1)
    return response['choices'][0]['message']['content']  # type: ignore


def get_utterance_by_TVID(msg_data: pd.DataFrame):
    for i in np.unique(msg_data["Speaker"]):
        tmp_a = msg_data[msg_data["Speaker"] == i]  # 获取当前的对话ID所对应样本

        comm = ""  # 提供的
        gpt = ""  # 输出的
        for v in np.unique(tmp_a["Speaker"]):
            gpt += "{}: {}\n".format(v, "神经质 {}, 外倾性 {}, 开放性 {}, 亲和性 {}, 尽责性 {}".format(
                np.unique(tmp_a[tmp_a["Speaker"] == v]["Neuroticism"])[0],
                np.unique(tmp_a[tmp_a["Speaker"] == v]["Extraversion"])[0],
                np.unique(tmp_a[tmp_a["Speaker"] == v]["Openness"])[0],
                np.unique(tmp_a[tmp_a["Speaker"] == v]["Agreeableness"])[0],
                np.unique(tmp_a[tmp_a["Speaker"] == v]["Conscientiousness"])[0]
            ))
            
        for j, row in tmp_a.iterrows():
            comm += "{}: {}\n".format(row["Speaker"], row["Utterance"])
        yield (comm, gpt)
        

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
