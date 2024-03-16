import os
from PIL import Image

rootdir = 'data/data_flow_scene'

count = 0
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        path = os.path.join(subdir, file)
        if file.endswith('.png'):
            img_png = Image.open(path)
            path_jpg = path.replace('.png', '.jpg')
            img_png.save(path_jpg)
            #os.remove(path)
            count+=1
            if count%100 == 0:
                print(count)
