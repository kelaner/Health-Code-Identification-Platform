import os


class Config:
    def __init__(self):
        pass

    src = "./output/1/"


def clean_temp():
    for i in os.listdir(Config.src):
        path = os.path.join(Config.src, i)
        os.remove(path)


if __name__ == '__main__':
    clean_temp()
