import os

if __name__ == "__main__":
    ROOT_DIR = input("폴더의 디렉토리 입력 : ")
    print(len(os.listdir(ROOT_DIR)))