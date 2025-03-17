import chardet

with open("/home/elwood/ds_new/cn30_attack.csv", "rb") as f:
    result = chardet.detect(f.read())
    print("Encoding:", result["encoding"])

with open("/home/elwood/ds_new/cn30_attack.csv", "r", encoding=result["encoding"], errors="ignore") as f:
    content = f.read()

with open("data_utf8.csv", "w", encoding="utf-8") as f:
    f.write(content)

print("Save as data_utf8.csv")
