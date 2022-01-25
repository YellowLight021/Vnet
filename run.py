# coding=utf-8

###### 欢迎使用脚本任务,让我们首选熟悉下一些使用规则吧 ######

# 数据集文件目录
datasets_prefix = '/root/paddlejob/workspace/train_data/datasets/'

# 数据集文件具体路径请在编辑项目状态下,通过左侧导航栏「数据集」中文件路径拷贝按钮获取
# train_datasets =  '通过路径拷贝获取真实数据集文件路径 '

# 输出文件目录. 任务完成后平台会自动把该目录所有文件压缩为tar.gz包，用户可以通过「下载输出」可以将输出信息下载到本地.
output_dir = "/root/paddlejob/workspace/output"

# 日志记录. 任务会自动记录环境初始化日志、任务执行日志、错误日志、执行脚本中所有标准输出和标准出错流(例如print()),用户可以在「提交」任务后,通过「查看日志」追踪日志信息.
import os

if __name__ == "__main__":
    print(os.listdir)
    print(os.getcwd())
    os.system('pip install -r requirements.txt')
    print('解压数据集')
    os.system("mkdir imgs")
    from unrar import rarfile
    file = rarfile.RarFile("{}/data118869/seg-lungs-LUNA16.rar".format(datasets_prefix))
    file.extractall("./")
    subsetnums=1
    for i in range(subsetnums):
        file = rarfile.RarFile("{}/data119126/subset{}.rar".format(datasets_prefix,i))
        file.extractall("./")
        os.system("mv ./subset{}/* ./imgs".format(i))

    #四卡训练
    os.system("export CUDA_VISIBLE_DEVICES=0,1,2,3")
    os.system("python train.py")

