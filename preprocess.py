import shutil
import warnings
import tables
import os
import SimpleITK as sitk
import numpy as np
from nipype.interfaces.ants import N4BiasFieldCorrection


def correct_bias(in_file, out_file, image_type=sitk.sitkFloat64):
    """
    Corrects the bias using ANTs N4BiasFieldCorrection. If this fails, will then attempt to correct bias using SimpleITK
    :param in_file: nii文件的输入路径
    :param out_file: 校正后的文件保存路径名
    :return: 校正后的nii文件全路径名
    """
    # 使用N4BiasFieldCorrection校正MRI图像的偏置场
    correct = N4BiasFieldCorrection()
    correct.inputs.input_image = in_file
    correct.inputs.output_image = out_file
    try:
        done = correct.run()
        return done.outputs.output_image
    except IOError:
        warnings.warn(RuntimeWarning("ANTs N4BIasFieldCorrection could not be found."
                                     "Will try using SimpleITK for bias field correction"
                                     " which will take much longer. To fix this problem, add N4BiasFieldCorrection"
                                     " to your PATH system variable. (example: EXPORT PATH=${PATH}:/path/to/ants/bin)"))
        input_image = sitk.ReadImage(in_file, image_type)
        output_image = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
        sitk.WriteImage(output_image, out_file)
        return os.path.abspath(out_file)


def normalize_image(in_file, out_file, bias_correction=True):
    # bias_correction：是否需要校正
    if bias_correction:
        correct_bias(in_file, out_file)
    else:
        shutil.copy(in_file, out_file)
    return out_file





def write_data_to_file(training_data_files, out_file, image_shape, truth_dtype=np.uint8, subject_ids=None,
                       normalize=True, crop=True):
    """
    Takes in a set of training images and writes those images to an hdf5 file.
    :param training_data_files: 训练的MRI文件列表，每一项是一个元组，元组中包括4种不同模态图像+ground truth的全路径
    Example: [('sub1-T1.nii.gz', 'sub1-T2.nii.gz', 'sub1-flair.nii.gz','sub1-t1ce.nii.gz','sub1-truth.nii.gz'),
              ('sub2-T1.nii.gz', 'sub2-T2.nii.gz', 'sub2-flair.nii.gz','sub2-t1ce.nii.gz' 'sub2-truth.nii.gz')]
    :param out_file: 保存的hdf5文件路径
    :param image_shape: 保存的image shape.
    :param truth_dtype: ground truth数据类型,Default is 8-bit unsigned integer.
    :return: Location of the hdf5 file with the image data written to it.
    """
    # 样本数
    n_samples = len(training_data_files)
    # 通道数等于模态数4
    n_channels = len(training_data_files[0]) - 1

    try:
        # 创建hdf5文件,在hdf5_file.root下创建3个压缩的可扩展数组
        hdf5_file, data_storage, truth_storage, affine_storage = create_data_file(out_file,
                                                                                  n_channels=n_channels,
                                                                                  n_samples=n_samples,
                                                                                  image_shape=image_shape)
    except Exception as e:
        # If something goes wrong, delete the incomplete data file
        os.remove(out_file)
        raise e

    # crop=True:裁剪
    write_image_data_to_file(training_data_files, data_storage, truth_storage, image_shape,
                             truth_dtype=truth_dtype, n_channels=n_channels, affine_storage=affine_storage, crop=crop)
    if subject_ids:
        hdf5_file.create_array(hdf5_file.root, 'subject_ids', obj=subject_ids)
    # 训练样本数据归一化
    if normalize:
        normalize_data_storage(data_storage)
    hdf5_file.close()
    # 返回hdf5文件名
    return out_file


def create_data_file(out_file, n_channels, n_samples, image_shape):
    # 创建hdf5文件
    hdf5_file = tables.open_file(out_file, mode='w')
    # 压缩可扩展数组
    # 定义一个filter来说明压缩方式及压缩深度
    filters = tables.Filters(complevel=5, complib='blosc')

    # 定义data和truth的shape
    ##能够扩展其一个维度,我们把可以扩展这个维度的shape设置为0
    data_shape = tuple([0, n_channels] + list(image_shape))  # (0,4,144,144,144)
    truth_shape = tuple([0, 1] + list(image_shape))  # (0,1,144,144,144)

    # 创建3个压缩可扩展数组保存data，truth和affine
    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape,
                                           filters=filters, expectedrows=n_samples)
    truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.UInt8Atom(), shape=truth_shape,
                                            filters=filters, expectedrows=n_samples)
    # 放射变换数组
    affine_storage = hdf5_file.create_earray(hdf5_file.root, 'affine', tables.Float32Atom(), shape=(0, 4, 4),
                                             filters=filters, expectedrows=n_samples)
    return hdf5_file, data_storage, truth_storage, affine_storage


def write_image_data_to_file(image_files, data_storage, truth_storage, image_shape, n_channels, affine_storage,
                             truth_dtype=np.uint8, crop=True):
    for set_of_files in image_files:
        # set_of_files：不同模态图像路径的元组('sub1-T1.nii.gz', 'sub1-T2.nii.gz', 'sub1-flair.nii.gz','sub1-t1ce.nii.gz','sub1-truth.nii.gz')
        # crop=True
        # 对4个模态+truth图像根据前景背景裁剪
        images = reslice_image_set(set_of_files, image_shape, label_indices=len(set_of_files) - 1, crop=crop)

        # 获取4个模态+truth的image的数组
        subject_data = [image.get_data() for image in images]

        # images[0].affine：4*4
        add_data_to_storage(data_storage, truth_storage, affine_storage, subject_data, images[0].affine, n_channels,
                            truth_dtype)
    # 返回data和ground_truth
    return data_storage, truth_storage


def add_data_to_storage(data_storage, truth_storage, affine_storage, subject_data, affine, n_channels, truth_dtype):
    # 添加1份subject_data数据，写入时将subject_data扩展到与create_data_file中定义的维度相同
    data_storage.append(np.asarray(subject_data[:n_channels])[
                            np.newaxis])  # np.asarray:==>[4,144,144,144] 扩展=new.axis:[1,4,144,144,144]
    truth_storage.append(np.asarray(subject_data[n_channels], dtype=truth_dtype)[np.newaxis][
                             np.newaxis])  # np.asarray:==>[144,144,144] 扩展=new.axis:[1,1,144,144,144]
    affine_storage.append(np.asarray(affine)[np.newaxis])  # np.asarray:==>[4,4] 扩展=new.axis:[1,4,,4]

def reslice_image_set(in_files, image_shape, out_files=None, label_indices=None, crop=False):
    #in_files:('sub1-T1.nii.gz', 'sub1-T2.nii.gz', 'sub1-flair.nii.gz','sub1-t1ce.nii.gz','sub1-truth.nii.gz')
    #label_indices:模态个数-4
    #对图像进行裁剪
    if crop:
        #返回各个维度要裁剪的范围[slice(),slice(),slice()]
        crop_slices = get_cropping_parameters([in_files])
    else:
        crop_slices = None
    #对in_files中的每个image裁剪放缩后返回的image列表
    images = read_image_files(in_files, image_shape=image_shape, crop=crop_slices, label_indices=label_indices)
    if out_files:
        for image, out_file in zip(images, out_files):
            image.to_filename(out_file)
        return [os.path.abspath(out_file) for out_file in out_files]
    else:
        return images

def get_cropping_parameters(in_files):
    #in_files:[('sub1-T1.nii.gz', 'sub1-T2.nii.gz', 'sub1-flair.nii.gz','sub1-t1ce.nii.gz','sub1-truth.nii.gz')]
    if len(in_files) > 1:
        foreground = get_complete_foreground(in_files)
    else:
        #return_image=True：返回foreground image
        #in_files[0]：('sub1-T1.nii.gz', 'sub1-T2.nii.gz', 'sub1-flair.nii.gz','sub1-t1ce.nii.gz','sub1-truth.nii.gz')
        foreground = get_foreground_from_set_of_files(in_files[0], return_image=True)
    #根据foreground确定各个维度要裁剪的范围，crop_img_to进行裁剪
    #return_slices=True：返回各个维度要裁剪的范围[slice(),slice(),slice()]
    return crop_img(foreground, return_slices=True, copy=True)


def get_foreground_from_set_of_files(set_of_files, background_value=0, tolerance=0.00001, return_image=False):
    # set_of_files：('sub1-T1.nii.gz', 'sub1-T2.nii.gz', 'sub1-flair.nii.gz','sub1-t1ce.nii.gz','sub1-truth.nii.gz')
    # image_file:sub1-T1.nii.gz
    for i, image_file in enumerate(set_of_files):
        # 读取image无裁剪，无resize
        image = read_image(image_file)
        # 根据设置的background值,综合所有模态和truth将image data数组中foreground位置设为True
        is_foreground = np.logical_or(image.get_data() < (background_value - tolerance),
                                      image.get_data() > (background_value + tolerance))
        if i == 0:
            foreground = np.zeros(is_foreground.shape, dtype=np.uint8)

        # 将is_foreground位置像素值设置为1
        foreground[is_foreground] = 1
    # 返回image
    if return_image:
        return new_img_like(image, foreground)
    # 返回数组
    else:
        return foreground


def read_image(in_file, image_shape=None, interpolation='linear', crop=None):
    print("Reading: {0}".format(in_file))
    image = nib.load(os.path.abspath(in_file))
    image = fix_shape(image)
    # 裁剪
    if crop:
        # crop example:[(45, 192, None), (47, 210, None), (0, 137, None)]分别代表X,Y,Z的裁剪坐标范围
        image = crop_img_to(image, crop, copy=True)
    # 放缩
    if image_shape:
        return resize(image, new_shape=image_shape, interpolation=interpolation)
    else:
        return image


def fix_shape(image):
    if image.shape[-1] == 1:
        # np.squeeze减少维度
        # image.__class__():根据data数组新建nii image
        return image.__class__(dataobj=np.squeeze(image.get_data()), affine=image.affine)
    return image


def crop_img(img, rtol=1e-8, copy=True, return_slices=False):
    img = check_niimg(img)
    data = img.get_data()

    infinity_norm = max(-data.min(), data.max())
    passes_threshold = np.logical_or(data < -rtol * infinity_norm,
                                     data > rtol * infinity_norm)

    if data.ndim == 4:
        passes_threshold = np.any(passes_threshold, axis=-1)
    coords = np.array(np.where(passes_threshold))
    start = coords.min(axis=1)
    end = coords.max(axis=1) + 1

    # pad with one voxel to avoid resampling problems
    start = np.maximum(start - 1, 0)
    end = np.minimum(end + 1, data.shape[:3])

    slices = [slice(s, e) for s, e in zip(start, end)]

    if return_slices:
        return slices

    return crop_img_to(img, slices, copy=copy)

def resize(image, new_shape, interpolation="linear"):
    image = reorder_img(image, resample=interpolation)
    zoom_level = np.divide(new_shape, image.shape)
    new_spacing = np.divide(image.header.get_zooms(), zoom_level)
    new_data = resample_to_spacing(image.get_data(), image.header.get_zooms(), new_spacing,
                                   interpolation=interpolation)
    new_affine = np.copy(image.affine)
    np.fill_diagonal(new_affine, new_spacing.tolist() + [1])
    new_affine[:3, 3] += calculate_origin_offset(new_spacing, image.header.get_zooms())
    return new_img_like(image, new_data, affine=new_affine)


def normalize_data(data, mean, std):
    # data：[4,144,144,144]
    data -= mean[:, np.newaxis, np.newaxis, np.newaxis]
    data /= std[:, np.newaxis, np.newaxis, np.newaxis]
    return data


def normalize_data_storage(data_storage):
    means = list()
    stds = list()
    # [n_example,4,144,144,144]
    for index in range(data_storage.shape[0]):
        # [4,144,144,144]
        data = data_storage[index]
        # 分别求出每个模态的均值和标准差
        means.append(data.mean(axis=(1, 2, 3)))
        stds.append(data.std(axis=(1, 2, 3)))
    # 求每个模态在所有样本上的均值和标准差[n_example,4]==>[4]
    mean = np.asarray(means).mean(axis=0)
    std = np.asarray(stds).mean(axis=0)
    for index in range(data_storage.shape[0]):
        # 根据均值和标准差对每一个样本归一化
        data_storage[index] = normalize_data(data_storage[index], mean, std)
    return data_storage

# training_files：训练的MRI文件列表，每一项是一个元组，元组中包括4种不同模态图像+ground truth的全路径
# config["image_shape"]：处理后的图片大小(144,144,144)
write_data_to_file(training_files, config["data_file"], image_shape=config["image_shape"])