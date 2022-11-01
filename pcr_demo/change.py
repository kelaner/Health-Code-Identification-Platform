import os
import shutil

root_dir = "./output/temp/"
list_i = os.listdir(root_dir)

for i in os.listdir(root_dir):
    full_path = os.path.join(root_dir, i)
    aim_dir = "./output/1"
    if not os.path.exists(aim_dir):
        os.mkdir(aim_dir)

    shutil.move(full_path, aim_dir)
