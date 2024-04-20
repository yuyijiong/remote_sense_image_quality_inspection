import json

json_path="swin_9label/config.json"
#读取json
with open(json_path, 'r') as f:
    data = json.load(f)
    print(data)

#保存json，ensure_ascii=False保证中文不乱码
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

