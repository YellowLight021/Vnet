import os
import paddle
import vnet
from glob import glob
import numpy as np
from luna import resampleVolume, truncate
import SimpleITK as sitk
import paddle.vision.transforms as transforms

MIN_BOUND = -1000
MAX_BOUND = 400
# target_shape=[64,128,128]
target_shape = [64, 128, 128]
target_spatial_resoluton = [1, 1, 1.5]
normMu = [-300]
normSigma = [700]
normTransform = transforms.Normalize(normMu, normSigma)
testTransform = transforms.Compose([
    # transforms.ToTensor(),
    normTransform
])


def doinfer(model, img):
    z, y, x = img.shape
    img = paddle.to_tensor(img)
    # import pdb
    # pdb.set_trace()
    if img.shape[0] < target_shape[0]:
        padding_zero = paddle.zeros([target_shape[0] - img.shape[0], target_shape[1], target_shape[2]],
                                    dtype='float32')
        img = paddle.concat([img, padding_zero], axis=0)

    data = testTransform(img)
    data = data.unsqueeze(0).unsqueeze(0)
    # import pdb
    # pdb.set_trace()
    output = model(data)
    # import pdb
    # pdb.set_trace()
    result = output.argmax(1)
    result = result[0, :z, :y, :x]
    # print(result[0, :z, :y, :x].shape)

    return result


def get_mask(model, img_file):
    itk_img = sitk.ReadImage(img_file)
    newvol = resampleVolume(target_spatial_resoluton, itk_img)
    img = sitk.GetArrayFromImage(newvol)
    # import pdb
    # pdb.set_trace()
    img = truncate(img, MIN_BOUND, MAX_BOUND)
    img = img.astype(np.float32)
    # img.dtype='float32'
    z, y, x = np.shape(img)
    # img = img.reshape((1, z, y, x))
    result_mask = np.zeros([z, y, x], dtype='int32')
    z_target, y_target, x_target = target_shape
    for z_start in range(0, z, z_target):
        if (z_start + z_target > z):
            z_start = z - z_target
        for y_start in range(0, y, y_target):
            if (y_start + y_target > y):
                y_start = y - y_target
            for x_start in range(0, x, x_target):
                if (x_start + x_target > x):
                    x_start = x - x_target
                inferimg = img[max(z_start, 0):(z_start + z_target), max(y_start, 0):(y_start + y_target),
                           max(x_start, 0):(x_start + x_target)]
                # import pdb
                # pdb.set_trace()
                result_mask[max(z_start, 0):(z_start + z_target), max(y_start, 0):(y_start + y_target),
                max(x_start, 0):(x_start + x_target)] = \
                    doinfer(model, inferimg)

    return result_mask


if __name__ == "__main__":
    model = vnet.VNet()
    params_path = "weights/checkpoint_model_best.pth.tar"
    checkpoint = paddle.load(params_path)
    model.set_state_dict(checkpoint['state_dict'])
    test_path = "imgs"
    test_result_path = "result_path"
    image_files = [file for file in os.listdir(test_path) if file.endswith(".mhd")]
    model.eval()
    for image_file in image_files:
        file_mask = get_mask(model, os.path.join(test_path, image_file))
        savename = image_file.replace(".mhd", "npy")
        np.save(os.path.join(test_result_path, savename), file_mask)





