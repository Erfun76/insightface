import os
import pathlib
import random

root = "persian_celeb_112x112"
for path, subdirs, files in os.walk(root):
    i = 1
    for name in files:
        extension = name.split(".")[-1].lower()
        # if extension != "jpg":
        #     continue
        '''
        if i == 0:
            stri = '00'
        elif 0 < i < 10:
            stri = '0' + str(i)
        else:
            stri = str(i)
        '''
        stri = str(random.randint(1,1000))
        os.rename(os.path.join(path, name), os.path.join(path, os.path.basename(path) + "_" + stri + '.' + extension))
        i = i + 1
        print(os.path.basename(path))
