# class LUNA16(data.Dataset):
import numpy as np
import paddle.dataset
from glob import glob
import os
import SimpleITK as sitk
from paddle.io import Dataset
import utils

MIN_BOUND = -1000
MAX_BOUND = 400
# target_shape=[64,128,128]
target_shape = [64, 128, 128]
target_spatial_resoluton = [1, 1, 1.5]
image_dict = {}
label_dict = {}
mask_dict = {}
stats_dict = {}
test_split = []
train_split = []


def resampleVolume(outspacing, vol):
    """
    将体数据重采样的指定的spacing大小\n
    paras：
    outpacing：指定的spacing，例如[1,1,1]
    vol：sitk读取的image信息，这里是体数据\n
    return：重采样后的数据
    """
    outsize = [0, 0, 0]
    inputspacing = 0
    inputsize = 0
    inputorigin = [0, 0, 0]
    inputdir = [0, 0, 0]

    # 读取文件的size和spacing信息

    inputsize = vol.GetSize()
    inputspacing = vol.GetSpacing()

    transform = sitk.Transform()
    transform.SetIdentity()
    # 计算改变spacing后的size，用物理尺寸/体素的大小
    outsize[0] = int(inputsize[0] * inputspacing[0] / outspacing[0] + 0.5)
    outsize[1] = int(inputsize[1] * inputspacing[1] / outspacing[1] + 0.5)
    outsize[2] = int(inputsize[2] * inputspacing[2] / outspacing[2] + 0.5)

    # 设定重采样的一些参数
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(vol.GetOrigin())
    resampler.SetOutputSpacing(outspacing)
    resampler.SetOutputDirection(vol.GetDirection())
    resampler.SetSize(outsize)
    newvol = resampler.Execute(vol)
    return newvol


def train_test_split(full, positive, test_fraction):
    negative = full - positive
    test_neg_count = int(np.ceil(len(negative) * test_fraction))
    test_pos_count = int(np.ceil(len(positive) * test_fraction))
    negative_list = list(negative)
    positive_list = list(positive)
    np.random.shuffle(positive_list)
    np.random.shuffle(negative_list)
    test_positive = set()
    for i in range(test_pos_count):
        test_positive |= set([positive_list[i]])
    train_positive = positive - test_positive
    if test_neg_count > 1:
        test_negative = set()
        for i in range(test_neg_count):
            test_negative |= set([negative_list[i]])
        train_negative = negative - test_negative
        train = list(train_positive | train_negative)
        test = list(test_positive | test_negative)
    else:
        train = list(train_positive)
        test = list(test_positive)
    np.random.shuffle(train)
    np.random.shuffle(test)
    return (train, test)


def truncate(image, min_bound, max_bound):
    image[image < min_bound] = min_bound
    image[image > max_bound] = max_bound
    return image


def load_image(root, series):
    if series in image_dict.keys():
        return image_dict[series]
    img_file = os.path.join(root, series + ".mhd")
    itk_img = sitk.ReadImage(img_file)
    newvol = resampleVolume(target_spatial_resoluton, itk_img)
    img = sitk.GetArrayFromImage(newvol)
    z, y, x = np.shape(img)
    img = img.reshape((1, z, y, x))
    # print('img_file:{},img.shape:{}'.format(img_file,img.shape))
    image_dict[series] = truncate(img, MIN_BOUND, MAX_BOUND)
    stats_dict[series] = itk_img.GetOrigin(), itk_img.GetSpacing()
    return img


def load_label(root, series):
    if series in label_dict.keys():
        return label_dict[series]
    img_file = os.path.join(root, series + ".mhd")
    itk_img = sitk.ReadImage(img_file)
    newvol = resampleVolume(target_spatial_resoluton, itk_img)
    img = sitk.GetArrayFromImage(newvol)
    if np.max(img) > 3400:
        img[img <= 3480] = 0
        img[img > 3480] = 1
    else:
        img[img != 0] = 1
    label_dict[series] = img.astype(np.uint8)
    return img


def load_mask(root, series):
    if series in mask_dict.keys():
        return mask_dict[series]
    img_file = os.path.join(root, series + ".mhd")
    itk_img = sitk.ReadImage(img_file)
    newvol = resampleVolume(target_spatial_resoluton, itk_img)
    img = sitk.GetArrayFromImage(newvol)
    img[img != 0] = 1
    mask_dict[series] = img
    return img


def full_dataset(dir, images):
    image_path = os.path.join(dir, images)
    image_files = glob(os.path.join(image_path, "*.mhd"))
    image_list = []
    for name in image_files:
        image_list.append(os.path.basename(name)[:-4])
    return image_list


# 原论文是mri图像需要N4矫正，当前实验为CT图像不确定是否需要矫正，那就暂时不写N4的处理代码
def make_dataset(dir, images, targets, seed, train, class_balance, partition, nonempty, test_fraction, mode):
    global image_dict, label_dict, test_split, train_split
    zero_tensor = None

    train = mode == "train"
    label_path = os.path.join(dir, targets)
    label_files = glob(os.path.join(label_path, "*.mhd"))
    # image_files=glob(os.path.join(os.path.join(dir, images), "*.mhd"))
    label_list = []
    for name in label_files:
        label_list.append(os.path.basename(name)[:-4])
    # for name in image_files:
    #     label_list.append(os.path.basename(name)[:-4])
    print("label_list.len:{}".format(len(label_list)))
    if len(test_split) == 0:
        zero_tensor = np.zeros(target_shape, dtype=np.uint8)
        image_list = []
        image_path = os.path.join(dir, images)
        file_list = glob(image_path + "/*.mhd")
        # import pdb
        # pdb.set_trace()
        for img_file in file_list:
            series = os.path.basename(img_file)[:-4]
            if series not in label_list:
                continue
            image_list.append(series)
            if series not in label_list:
                label_dict[series] = zero_tensor
        np.random.seed(seed)
        full = set(image_list)
        positives = set(label_list) & full
        train_split, test_split = train_test_split(full, positives, test_fraction)
        print("train_split:{}".format(len(train_split)))
        print("test_split:{}".format(len(test_split)))
    if train:
        keys = train_split
    else:
        keys = test_split
    # import pdb
    # pdb.set_trace()
    result = []
    # target_means = []
    for index in range(len(keys)):

        sample_label = load_label(label_path, keys[index])
        shape = np.shape(sample_label)

        part_list = []
        z, y, x = shape
        if target_shape is not None:
            z_target, y_target, x_target = target_shape
            z, y, x = shape
            for z_start in range(0, z, z_target):
                if (z_start + z_target > z):
                    z_start = z - z_target
                for y_start in range(0, y, y_target):
                    if (y_start + y_target > y):
                        y_start = y - y_target
                    for x_start in range(0, x, x_target):
                        if (x_start + x_target > x):
                            x_start = x - x_target
                        part_list.append(((max(z_start, 0), z_start + z_target), (max(y_start, 0), y_start + y_target),
                                          (max(x_start, 0), x_start + x_target)))
        else:
            part_list = [((0, z), (0, y), (0, x))]
        for part in part_list:
            if nonempty:
                if np.sum(utils.get_subvolume(sample_label, part)) == 0:
                    continue
            # target_means.append(np.mean(sample_label))
            result.append((keys[index], part))
    # target_mean = np.mean(target_means)
    return result


class LUNA16(Dataset):
    def __init__(self, root='.', images=None, targets=None, transform=None, mode="train", seed=1,
                 class_balance=False, split=None, masks=None, nonempty=True,
                 test_fraction=0.2):
        if images is None:
            raise (RuntimeError("images must be set"))
        if targets is None and mode != "infer":
            raise (RuntimeError("both images and targets must be set if mode is not 'infer'"))
        if mode == "infer":
            imgs = full_dataset(root, images)
        else:
            imgs = make_dataset(root, images, targets, seed, mode, class_balance, split, nonempty,
                                test_fraction, mode)
            # import pdb
            # pdb.set_trace()
            # self.data_mean = target_mean
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images: " + os.path.join(root, images) + "\n"))

        self.mode = mode
        self.root = root
        self.imgs = imgs
        self.masks = None
        self.split = split
        if masks is not None:
            self.masks = os.path.join(self.root, masks)
        if targets is not None:
            self.targets = os.path.join(self.root, targets)
        self.images = os.path.join(self.root, images)
        self.transform = transform

        # import pdb
        # pdb.set_trace()

    def target_mean(self):
        return self.data_mean

    def __getitem__(self, index):
        if self.mode == "train" or self.mode == "eval":
            return self.__getitem_dev(index)
        elif self.mode == "infer":
            return self.__getitem_prod(index)

    def __getitem_prod(self, index):
        series = self.imgs[index]
        image = load_image(self.images, series)
        origin, spacing = stats_dict[series]
        image = image.astype(np.float32)
        if self.split is not None:
            batches = utils.partition_image(image, self.split)
        else:
            batches = [image]
        if self.transform is not None:
            batches = map(self.transform, batches)
            batches = [*batches]
        # batches = torch.cat(batches)
        batches = paddle.concat(batches)
        return batches, series, origin, spacing

    def __getitem_dev(self, index):
        series, bounds = self.imgs[index]
        # print('series:{},bounds:{}'.format(series,bounds))
        (zs, ze), (ys, ye), (xs, xe) = bounds
        # import pdb
        # pdb.set_trace()
        target = load_label(self.targets, series)
        target = target[zs:ze, ys:ye, xs:xe]

        # target = torch.from_numpy(target.astype(np.int64))

        image = load_image(self.images, series)
        # print("image.size{},image.shape{}".format(image.size,image.shape))
        image = image[0, zs:ze, ys:ye, xs:xe]
        # optionally mask out lung volume from image
        # import pdb
        # pdb.set_trace()

        if self.masks is not None:
            mask = load_mask(self.masks, series)
            mask = mask[zs:ze, ys:ye, xs:xe]
            image -= MIN_BOUND
            image = np.multiply(mask, image)
            image += MIN_BOUND
        # image = image.reshape((1, ze-zs, ye-ys, xe-xs))
        image = image.reshape((ze - zs, ye - ys, xe - xs))
        image = image.astype(np.float32)
        # 这里进行deform变换处理，只给训练的时deform变幻，测试的时候不做数据增强变幻
        if self.mode == "train":
            if np.random.rand(1)[0] > 0.3:
                img, target = utils.produceRandomlyDeformedImage(image, target, 2, 15)
                # print("用了deform。。。。。。。。")
                if np.isnan(img).any():
                    print("抓到一个变换后的脏数据")
                    img = image
            else:
                img = image
        else:
            img = image

        img = truncate(img, MIN_BOUND, MAX_BOUND)
        img = paddle.to_tensor(img)
        target = paddle.to_tensor(target.astype(np.int64))
        # import pdb
        # pdb.set_trace()
        if img.shape[0] < target_shape[0]:
            padding_zero = paddle.zeros([target_shape[0] - img.shape[0], target_shape[1], target_shape[2]],
                                        dtype='float32')
            img = paddle.concat([img, padding_zero], axis=0)
        if self.mode == 'train':
            padding_zero = paddle.zeros([target_shape[0] - target.shape[0], target_shape[1], target_shape[2]],
                                        dtype='int64')
            target = paddle.concat([target, padding_zero], axis=0)

        if self.transform is not None:
            img = self.transform(img)
            # target不需要归一化操作
            # target = self.transform(target)

        img = img.unsqueeze(0)

        return img, target

    def __len__(self):
        return len(self.imgs)
