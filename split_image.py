import cv2
import os
import pandas as pd
from tqdm import tqdm


def split_ldr(root, size):
    for file in tqdm(os.listdir(root)):

        imgroot = os.path.join(root, file)
        image = cv2.imread(imgroot)

        height = image.shape[0]
        width = image.shape[1]

        file = file.split('.')[0]
        for y in range(0, height, size):
            for x in range(0, width, size):
                tiles = image[y:y + size, x:x + size]
                if (tiles.shape[0] == size) and (tiles.shape[1] == size):
                    cv2.imwrite('data/train/split_img' + '/' + file + str(x) + '2' + str(y) + ".jpg", tiles)
    #os.remove(root + '/input_2_aligned.tif')


def split_hdr(root, size):
    for file in tqdm(os.listdir(root)):
        imgroot = os.path.join(root, file)
        image = cv2.imread(imgroot, cv2.IMREAD_ANYDEPTH)

        height = image.shape[0]
        width = image.shape[1]

        file = file.split('.')[0]
        for y in range(0, height, size):
            for x in range(0, width, size):
                tiles = image[y:y + size, x:x + size]
                if (tiles.shape[0] == size) and (tiles.shape[1] == size):
                    cv2.imwrite('data/train/split_img' + '/' + file + str(x) + '2' + str(y) + ".hdr", tiles)
        #os.remove(root + '/ref_hdr_aligned.hdr')


if __name__ == '__main__':
    #for file in tqdm(os.listdir('data/train/LDR')):
    #split_ldr('data/train/LDR', size=256)
    #for file in tqdm(os.listdir('data/train/HDR')):
    #split_hdr('data/train/HDR', size=256)

    df = pd.DataFrame()
    f = open('data/test/annotations.txt', 'w+')
    path = 'data/test/image_split'

    name_list = list()
    for file in os.listdir(path):
        name = file.split('.')[0]
        #name = name.split('_')[1]
        if name not in name_list:
            f.write(path + '/' + name + '.png' + '%%' + path + '/' + name + '.hdr\n')
            name_list.append(name)
    f.close()