import json


def jsonload(fname: str, encoding="utf-8"):
    """config.json 파일을 로드

    Args:
        fname (str): _description_
        encoding (str): Defaults to "utf-8".

    Returns:
        json
    """
    with open(fname, encoding=encoding) as f:
        j = json.load(f)

    return j


def jsonlload(fname: str, encoding="utf-8"):
    """jsonl 데이터 파일을 읽어서 리스트로 반환

    Args:
        fname (str): _description_
        encoding (str): Defaults to "utf-8".

    Returns:
        jsonl에 담긴 내용을 리스트로 반환
    """
    json_list = []

    with open(fname, encoding=encoding) as f:
        for line in f.readlines():
            json_list.append(json.loads(line))

    return json_list


# json 객체를 파일로 저장
def jsondump(j, fname):
    """json 객체를 파일로 저장

    Args:
        j (json): json 객체
        fname (str): 저장할 파일 명
    """
    with open(fname, "w", encoding="UTF8") as f:
        json.dump(j, f, ensure_ascii=False)
