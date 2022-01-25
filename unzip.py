# 解压img文件
import os
import zipfile

datasets_prefix = '/root/paddlejob/workspace/train_data/datasets/data1860/'
dataset_label = '/root/paddlejob/workspace/train_data/datasets/data119677/seg-lungs-LUNA16.zip'
zipfile.ZipFile(dataset_label).extractall('./labels')
# print("**********")
# print(os.listdir('./labels/seg-lungs-LUNA16'))
# print("**********")
ziplist = os.listdir(datasets_prefix)
# 有100g的限制所以不能全部解压缩
count = 0
max_count = 2
for f in ziplist:

    if count <= max_count:
        # if f!="luna16-3.zip":
        print(f)
        extracting = zipfile.ZipFile(os.path.join(datasets_prefix, f))
        extracting.extractall('./imgs')
        # print('================================')
        # print(os.listdir('./'))
        # print('================================')
        ziplist = [f for f in os.listdir('./imgs') if f.endswith('.zip')]
        for zf in ziplist:
            extracting = zipfile.ZipFile(os.path.join('./imgs', zf))
            extracting.extractall('./imgs')
            os.system("rm ./imgs/{}".format(zf))
        # print('================================')
        # print(os.listdir('./imgs'))
        # print('================================')
        # os.system("mv ./{}/* ./imgs".format(f[:-4]))
        count += 1
    else:
        break
# print('imgs:{}'.format(os.listdir('./imgs')))
print('len imgs :{}'.format(len(os.listdir('./imgs'))))