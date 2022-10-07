import json

# config.json 로드
def jsonload(fname, encoding="utf-8"):
    with open(fname, encoding=encoding) as f:
        j = json.load(f)

    return j


# jsonl 데이터 파일 읽어서 리스트에 저장
def jsonlload(fname, encoding="utf-8"):
    json_list = []

    with open(fname, encoding=encoding) as f:
        for line in f.readlines():
            json_list.append(json.loads(line))

    return json_list


# json 객체를 파일로 저장
def jsondump(j, fname):
    with open(fname, "w", encoding="UTF8") as f:
        json.dump(j, f, ensure_ascii=False)
