import os
import shutil
import zipfile
dataset='/root/paddlejob/workspace/train_data/datasets/data127509/prostate.zip'
zipfile.ZipFile(dataset).extractall('./Prostate')
train_data_list=['TrainingData_Part1','TrainingData_Part2','TrainingData_Part3']
os.makedirs('labels',exist_ok=True)
os.makedirs('imgs',exist_ok=True)
for data_path in train_data_list:
    for filename in os.listdir(os.path.join('./Prostate',data_path)):
        # print(filename)
        if "segmentation" in filename:
            if not filename.endswith("raw"):
                newfilename=filename.replace("_segmentation","")
            else:
                newfilename=filename
            shutil.move(os.path.join('./Prostate',data_path,filename),os.path.join('./labels',newfilename))
        else:
            shutil.move(os.path.join('./Prostate',data_path,filename),'./imgs')