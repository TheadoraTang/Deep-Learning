# 读取以gbk格式存储的文件
with open('predict.txt', 'r', encoding='gbk') as f:
    content = f.read()

# 将内容写入UTF-8格式的文件
with open('predict.txt', 'w', encoding='utf-8') as f:
    f.write(content)
