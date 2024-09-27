from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines import pipeline
import pandas as pd

# 上传文件
data_api = []
data = pd.read_csv("test.csv")
answers = []
labels = []
error_num = 0
count = 0
for i in range(200):
    data_api.append(data["text"][i])
    labels.append((data["label"][i]))
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="unslothgemma-2-2b-it-bnb-4bit")
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="unslothgemma-2-2b-it-bnb-4bit")
for i in range(len(data_api)):
    messages = [
        {
            "role": "user",
            "content": "You are an emotion classifier.Users will give you English sentences and you "
                       "will judge the emotion the sentences express,1 stands for good,0 stands for bad."
                       "You only need to response 1 or 0.",
        },
        {"role": "model", "content": "ok,fine"},
        {"role": "user", "content": data_api[i]},
    ]
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   return_tensors="pt")
    # 进行推理
    outputs = model.generate(input_ids=tokenized_chat)

    # 解码输出
    decoded_output = tokenizer.decode(outputs[0])  # skip_special_tokens 去除特殊标记
    # 找到最后一个模型的回复，过滤标记符
    decoded_output = decoded_output.split("<start_of_turn>")
    for one in reversed(decoded_output):
        if "model" in one:
            # 提取模型最后一轮的内容，去掉 "model" 标记和 "<end_of_turn>"
            last_response = one.replace("model", "").replace("<end_of_turn>", "").strip()
            print(last_response)
            break
    answers.append(last_response)
    if last_response == str(labels[i]):
        count = count + 1
    print(f"{i}/{len(labels)}")
accuracy = 0.0
accuracy = count / len(labels)
print(f"zero-shot accuracy:{accuracy}")
df = pd.DataFrame(data=answers, columns=["prediction(zero-shot)"])
# df.to_csv("prediction_zero-shot_Gemma.csv")
