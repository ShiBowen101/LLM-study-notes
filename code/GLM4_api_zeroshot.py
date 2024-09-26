from zhipuai import ZhipuAI
import pandas as pd

# 上传文件
data_api = []
data = pd.read_csv("train.csv")
answers = []
labels = []
error_num = 0
for i in range(200):
    data_api.append(data["text"][i])
    labels.append((data["label"][i]))
for i in range(len(data_api)):
    try:
        client = ZhipuAI(api_key="ad7659b2733636d3c5d338801d94601d.MDq36QCvkfEYsPGl")  # 填写您自己的APIKey
        response = client.chat.completions.create(
            model="glm-4-flash",  # 填写需要调用的模型编码
            messages=[
                {"role": "system",
                 "content": "Your are an emotion classifier.Users will give you English sentences and you "
                            "will judge the emotion the sentences express,1 stands for good,0 stands for bad."
                            "You only need to response 1 or 0."},
                {"role": "user",
                 "content": data_api[i]

                 },
            ],
        )
    except Exception as e:
        error_num = error_num + 1
        print("出现报错，跳过此句子")
        labels.pop(i)
        continue

    if response.choices[0].message.content != '0' and response.choices[0].message.content != '1':
        response.choices[0].message.content = '2'  # 如果回复其他内容的默认分类错误
        print(response.choices[0].message.content)
    print(f"{i}/{len(labels)}")
    answers.append(int(response.choices[0].message.content))
accuracy = 0.0
count = 0
for i in range(len(answers)):
    if answers[i] == labels[i]:
        count = count + 1
accuracy = count / len(answers)
print(accuracy)
df = pd.DataFrame(data=answers, columns=["prediction(zero-shot)"])
# df.to_csv("prediction_zero-shot.csv")
