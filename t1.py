import requests


def test1():
    "提交验证码图片文件"
    imname = "11216h_3579db1e15a3541dc5b696f6093e1cc4.png"
    files = {"file": open(imname, "rb")}
    r = requests.post("http://127.0.0.1:5000/upload", files=files)
    print(r.json())


def test2():
    "提交返回结果"
    reqid = "7537ea4b-b26d-4263-a628-6b02c2d37add"
    status = 1
    r = requests.get(
        "http://127.0.0.1:5000/upload", params={"reqid": reqid, "status": status}
    )
    print(r.json())


if __name__ == "__main__":
    test2()
