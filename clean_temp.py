import os


class Config:
    def __init__(self):
        pass

    src = "./output/temp/"


def clean_temp():
    for i in os.listdir(Config.src):
        path = os.path.join(Config.src, i)
        # noinspection PyBroadException
        try:
            os.remove(path)
        except Exception:
            pass


if __name__ == '__main__':
    clean_temp()
