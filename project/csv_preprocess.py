# 이 파일의 이름을 csv.py로 하지 마세요

from langchain_community.document_loaders.csv_loader import CSVLoader

# csv파일 불러오기
loader = CSVLoader(
    file_path="recipe2.csv",  # csv파일의 이름은 영어 이름을 쓰세요
    encoding="cp949",
)
data = loader.load()

print(data)

# 단순하게 쭉 써서 txt파일로 저장
f = open("recipe2.txt", "w")
for x in data:
    f.write(str(x))
