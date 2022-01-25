import SimpleITK as sitk
import paddle
import os.path
import matplotlib.pyplot as plt
# import torch.nn
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import scipy.ndimage
import paddle.nn.functional as F
import paddle.fluid as fluid
from paddle.fluid.layers import one_hot,reduce_sum,reduce_mean
Z_MAX = None
Y_MAX = None
X_MAX = None
vox_spacing = None
shape_max = None

def produceRandomlyTranslatedImage(image, label):
    sitkImage = sitk.GetImageFromArray(image, isVector=False)
    sitklabel = sitk.GetImageFromArray(label, isVector=False)

    itemindex = np.where(label > 0)
    randTrans = (0,np.random.randint(-np.min(itemindex[1])/2,(image.shape[1]-np.max(itemindex[1]))/2),np.random.randint(-np.min(itemindex[0])/2,(image.shape[0]-np.max(itemindex[0]))/2))
    translation = sitk.TranslationTransform(3, randTrans)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitkImage)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(translation)

    outimgsitk = resampler.Execute(sitkImage)
    outlabsitk = resampler.Execute(sitklabel)

    outimg = sitk.GetArrayFromImage(outimgsitk)
    outimg = outimg.astype(dtype=float)

    outlbl = sitk.GetArrayFromImage(outlabsitk) > 0
    outlbl = outlbl.astype(dtype=float)

    return outimg, outlbl

def produceRandomlyDeformedImage(image, label, numcontrolpoints, stdDef):
    sitkImage=sitk.GetImageFromArray(image, isVector=False)
    sitklabel=sitk.GetImageFromArray(label, isVector=False)

    transfromDomainMeshSize=[numcontrolpoints]*sitkImage.GetDimension()

    tx = sitk.BSplineTransformInitializer(sitkImage,transfromDomainMeshSize)


    params = tx.GetParameters()

    paramsNp=np.asarray(params,dtype=float)
    paramsNp = paramsNp + np.random.randn(paramsNp.shape[0])*stdDef

    paramsNp[0:int(len(params)/3)]=0 #remove z deformations! The resolution in z is too bad

    params=tuple(paramsNp)
    tx.SetParameters(params)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitkImage)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(tx)

    resampler.SetDefaultPixelValue(0)
    outimgsitk = resampler.Execute(sitkImage)
    outlabsitk = resampler.Execute(sitklabel)

    outimg = sitk.GetArrayFromImage(outimgsitk)
    outimg = outimg.astype(dtype=np.float32)

    outlbl = sitk.GetArrayFromImage(outlabsitk)
    outlbl = (outlbl>0.5).astype(dtype=np.float32)

    return outimg,outlbl

def error_rate(output,target):
    out=paddle.argmax(output,axis=1)
    error_rate=1-(1.0*paddle.sum(out==target))/target.size
    # import pdb
    # pdb.set_trace()
    return error_rate.item()



def paddle_dice_loss(output,target):
    output = output.transpose([0, 2, 3, 4, 1]).reshape([-1,2])
    # print(target.numel())
    target=paddle.cast(target.reshape((target.numel(),)),"float32")
    eps = 0.000001
    # eps=0.00001
    result_ = paddle.argmax(output,1)
    result_ = paddle.cast(paddle.squeeze(result_),"float32")
    # if input.is_cuda:
    #     result = torch.cuda.FloatTensor(result_.size())
    #     target_ = torch.cuda.FloatTensor(target.size())
    # else:
    #     result = torch.FloatTensor(result_.size())
    #     target_ = torch.FloatTensor(target.size())
    # result.copy_(result_.data)
    # target_.copy_(target.data)
    # target = target_
    # print('reusult.shape:{}'.format(result.shape))
    # print('target.shape:{}'.format(target.shape))
    intersect = paddle.dot(result_, target)

    result_sum = paddle.sum(result_)
    target_sum = paddle.sum(target)
    union = result_sum + target_sum + 2 * eps
    # print('intersect:{}'.format(intersect))
    # intersect = torch.max([eps, intersect])

    # the target volume can be empty - so we still want to
    # end up with a score of 1 if the result is 0/0
    IoU = intersect / union
    #    print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
    #        union, intersect, target_sum, result_sum, 2*IoU))
    return 2 * IoU

def my_dice_loss(output,target):
    output = output.transpose([0, 2, 3, 4, 1])
    target = target.unsqueeze(axis=1)
    target = target.transpose([0, 2, 3, 4, 1])
    return dice_loss(output, target)
def dice_loss(input, label, epsilon=0.000001, name=None):
    label = paddle.fluid.layers.one_hot(label, depth=input.shape[-1])
    reduce_dim = list(range(1, len(input.shape)))
    inse = reduce_sum(input * label, dim=reduce_dim)
    dice_denominator = reduce_sum(
        input*input, dim=reduce_dim) + reduce_sum(
            label*label, dim=reduce_dim)
    dice_score = 1 - inse * 2 / paddle.clip(dice_denominator,epsilon)
    return reduce_mean(dice_score)


def init_dims3D(z, y, x, spacing):
    global Z_MAX, Y_MAX, X_MAX, vox_spacing, shape_max
    vox_spacing = spacing
    Z_MAX, Y_MAX, X_MAX = z, y, x
    shape_max = (z, y, x)


def debug_img(img):
    plt.hist(img.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()


def plot_3d(image, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    # p = image

    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


def npz_save(name, obj):
    keys = list(obj.keys())
    values = list(obj.values())
    np.savez(name + ".npz", keys=keys, values=values)


def npz_save_compressed(name, obj):
    keys = list(obj.keys())
    values = list(obj.values())
    np.savez_compressed(name + "_compressed.npz", keys=keys, values=values)


def npz_load(filename):
    npzfile = np.load(filename + ".npz")
    keys = npzfile["keys"]
    values = npzfile["values"]
    return dict(zip(keys, values))


def npz_load_compressed(filename):
    npzfile = np.load(filename + "_compressed.npz")
    keys = npzfile["keys"]
    values = npzfile["values"]
    return dict(zip(keys, values))


def copy_slice_centered(dst, src, dim):
    if dim <= Y_MAX:
        x_start = int((X_MAX - dim) / 2)
        y_start = int((Y_MAX - dim) / 2)
        for y in range(dim):
            for x in range(dim):
                dst[y_start + y][x_start + x] = src[y][x]
    elif dim <= X_MAX:
        x_start = int((X_MAX - dim) / 2)
        y_start = int((dim - Y_MAX) / 2)
        for y in range(Y_MAX):
            for x in range(dim):
                dst[y][x_start + x] = src[y_start + y][x]
    else:
        x_start = int((dim - X_MAX) / 2)
        y_start = int((dim - Y_MAX) / 2)
        for y in range(Y_MAX):
            for x in range(X_MAX):
                dst[y][x] = src[y_start + y][x_start + x]


def copy_normalized(src, dtype=np.int16):
    src_shape = np.shape(src)
    if src_shape == shape_max:
        return src

    (z_axis, y_axis, x_axis) = src_shape
    print(src_shape)
    assert x_axis == y_axis
    new_img = np.full(shape_max, np.min(src), dtype=dtype)
    if z_axis < Z_MAX:
        start = int((Z_MAX - z_axis) / 2)
        for i in range(z_axis):
            copy_slice_centered(new_img[start + i], src[i], x_axis)
    else:
        start = int((z_axis - Z_MAX) / 2)
        for i in range(Z_MAX):
            copy_slice_centered(new_img[i], src[start + i], x_axis)
    return new_img


def truncate(image, min_bound, max_bound):
    image[image < min_bound] = min_bound
    image[image > max_bound] = max_bound
    return image


def resample_volume(img, spacing_old, spacing_new, bounds=None):
    (z_axis, y_axis, x_axis) = np.shape(img)
    print('img: {} old spacing: {} new spacing: {}'.format(np.shape(img), spacing_old, spacing_new))
    resize_factor = np.array(spacing_old) / spacing_new
    new_shape = np.round(np.shape(img) * resize_factor)
    real_resize_factor = new_shape / np.shape(img)
    img_rescaled = scipy.ndimage.interpolation.zoom(img, real_resize_factor, mode='nearest').astype(np.int16)
    img_array_normalized = copy_normalized(img_rescaled)
    img_tmp = img_array_normalized.copy()
    # determine what the mean will be on the anticipated value range
    mu, var = 0., 0.
    if bounds is not None:
        min_bound, max_bound = bounds
        img_tmp = truncate(img_tmp, min_bound, max_bound)
        mu = np.mean(img_tmp)
        var = np.var(img_tmp)
    return (img_array_normalized, mu, var)


def save_updated_image(img_arr, path, origin, spacing):
    itk_scaled_img = sitk.GetImageFromArray(img_arr, isVector=False)
    itk_scaled_img.SetSpacing(spacing)
    itk_scaled_img.SetOrigin(origin)
    sitk.WriteImage(itk_scaled_img, path)


def save_image(img_arr, path):
    itk_img = sitk.GetImageFromArray(img_arr, isVector=False)
    sitk.WriteImage(itk_img, path)


def get_subvolume(target, bounds):
    (zs, ze), (ys, ye), (xs, xe) = bounds
    return np.squeeze(target)[zs:ze, ys:ye, xs:xe]


def partition_image(image, partition):
    z_p, y_p, x_p = partition
    z, y, x = np.shape(np.squeeze(image))
    z_incr, y_incr, x_incr = z // z_p, y // y_p, x // x_p
    assert z % z_p == 0
    assert y % y_p == 0
    assert x % x_p == 0
    image_list = []
    for zi in range(z_p):
        zstart = zi * z_incr
        zend = zstart + z_incr
        for yi in range(y_p):
            ystart = yi * y_incr
            yend = ystart + y_incr
            for xi in range(x_p):
                xstart = xi * x_incr
                xend = xstart + x_incr
                subvolume = get_subvolume(image, ((zstart, zend), (ystart, yend), (xstart, xend)))
                subvolume = subvolume.reshape((1, 1, z_incr, y_incr, x_incr))
                image_list.append(subvolume)
    return image_list


def merge_image(image_list, partition):
    z_p, y_p, x_p = partition
    shape = np.array(np.shape(image_list[0]), dtype=np.int32)
    z, y, x = 0, 0, 0
    z, y, x = shape * partition
    i = 0
    z_list = []
    for zi in range(z_p):
        y_list = []
        for yi in range(y_p):
            x_list = []
            for xi in range(x_p):
                x_list.append(image_list[i])
                i += 1
            y_list.append(np.concatenate(x_list, axis=2))
        z_list.append(np.concatenate(y_list, axis=1))
    return np.concatenate(z_list)


# Load the scans in given folder path
# def dicom_load_scan(path):
#     attr = {}
#     slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
#     slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
#
#     slices2 = []
#     prev = -1000000
#     # remove redundant slices
#     for slice in slices:
#         cur = slice.ImagePositionPatient[2]
#         if cur == prev:
#             continue
#         prev = cur
#         slices2.append(slice)
#     slices = slices2
#
#     for i in range(len(slices) - 1):
#         try:
#             slice_thickness = np.abs(slices[i].ImagePositionPatient[2] - slices[i + 1].ImagePositionPatient[2])
#         except:
#             slice_thickness = np.abs(slices[i].SliceLocation - slices[i + 1].SliceLocation)
#         if slice_thickness != 0:
#             break
#
#     print('patient: {} slice: {}'.format(os.path.basename(path), slice_thickness))
#
#     assert slice_thickness != 0
#
#     for s in slices:
#         s.SliceThickness = slice_thickness
#
#     x, y = slices[0].PixelSpacing
#     attr['Spacing'] = (x, y, slice_thickness)
#     attr['Origin'] = slices[0].ImagePositionPatient
#
#     return (slices, attr)


# def dicom_get_pixels_hu(slices):
#     image = np.stack([s.pixel_array for s in slices])
#     image = image.astype(np.int16)
#
#     # Convert to Hounsfield units (HU)
#     for slice_number in range(len(slices)):
#
#         intercept = slices[slice_number].RescaleIntercept
#         slope = slices[slice_number].RescaleSlope
#
#         if slope != 1:
#             image[slice_number] = slope * image[slice_number].astype(np.float64)
#             image[slice_number] = image[slice_number].astype(np.int16)
#
#         image[slice_number] += np.int16(intercept)
#
#     return np.array(image, dtype=np.int16)


# def dicom_convert(src, dst):
#     for scandir in os.listdir(src):
#         slices, attr = dicom_load_scan(os.path.join(src, scandir))
#         image = dicom_get_pixels_hu(slices)
#         save_updated_image(image, os.path.join(dst, scandir + '.mhd'),
#                            attr['Origin'], attr['Spacing'])
