import requests

imname = "11216h_3579db1e15a3541dc5b696f6093e1cc4.png"

files = {"file": open(imname, "rb")}

r = requests.post("http://127.0.0.1:5000/upload", files=files)

print(r.text)
