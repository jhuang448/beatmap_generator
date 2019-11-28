import os
import numpy as np

def write_maplist(map_dir, level, maplist_dir, unzip=False):

    osz_dir = os.path.join(map_dir, 'osz')

    if unzip:
        # mkdir
        if os.path.exists(osz_dir) == False:
            os.mkdir(osz_dir)
        # unzip .rar
        files = os.listdir(map_dir)
        for name in files:
            if name.endswith('.rar'):
                os.system("unrar x -y '{}' '{}'".format(os.path.join(map_dir, name), osz_dir))
                os.system("mv '{}' ~/Trash/".format(os.path.join(map_dir, name)))

        # unzip .osz
        osz_files = os.listdir(osz_dir)
        for file in osz_files:
            id = file.split(' ')[0]
            name = file[:-4]
            folder = os.path.join(map_dir, name)
            if os.path.exists(folder) == False:
                os.mkdir(folder)
            file_f = os.path.join(osz_dir, file)
            cmd = "unzip -d '{}' -n '{}' '*.mp3'".format(folder, file_f)
            os.system(cmd)
            cmd = "unzip -d '{}' -n '{}' '*.osu'".format(folder, file_f)
            os.system(cmd)
            os.system("mv '{}' ~/Trash/".format(file_f))

    maplist_name = os.path.join(maplist_dir, "maplist_{}.txt".format(level))
    with open(maplist_name, 'w') as F:
        folders = os.listdir(map_dir)
        for folder in folders:
            if os.path.isdir(os.path.join(map_dir, folder)) == False:
                continue
            files = os.listdir(os.path.join(map_dir, folder))
            for file in files:
                if file.endswith('.osu') and level in file:
                    line = os.path.join(map_dir, folder, file) + '\n'
                    print(line)
                    F.write(line)
                    break # prevent multiple normals for each song

    return

if __name__ == "__main__":
    write_maplist(map_dir='./maps/', level='Easy', maplist_dir='./mapdata/', unzip=False)