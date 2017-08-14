import os

directory = './air_car'
folderlist = os.listdir(directory)
lable_arr = []

for folder in folderlist:
    try:
        filelist = os.listdir(directory + '/' + folder)
    except Exception as e:
        print(e)
        continue

    for filename in filelist:
        try:
            lable_arr.append(folder.encode('utf8'))
            print()
        except Exception as e:
            print(e)