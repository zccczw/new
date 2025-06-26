import torch
print(torch.__version__)# 学校：湖北工业大学
# 开发人员：Barbaric Growth
# 开发时间：2022/11/7 16:18
labels = [1,2,3,4,5,6]
text_labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
    'horse', 'ship', 'truck']
a = [text_labels[int(i)] for i in labels]
print(a)