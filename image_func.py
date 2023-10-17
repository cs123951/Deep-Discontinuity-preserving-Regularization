# -*- coding: utf-8 -*-
# @Time    : 2021/3/13
# @Author  : Jiayi Lu
# @Python  : 3.7.4
# @File    : image_func.py

import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

import matplotlib.pyplot as plt
from matplotlib import gridspec


def disp_filter(disp01, filter):
    # disp01: batch, 3, x, y, z, in gpu, tensor.
    # batch equals to 1
    device = disp01.device
    if filter == "gaussian":
        for i in range(3):
            disp01 = gaussian_filter(disp01[0].cpu().numpy())
            disp01 = torch.tensor(disp01, device=device).unsqueeze(0)
    elif filter == "anisotropic":
        for i in range(1):
            disp01 = anisotropic_filter(disp01[0]).unsqueeze(0)
    return disp01


def gaussian_filter(disp):
    sigma = 1
    x1 = ndi.gaussian_filter(disp[0], sigma=sigma)
    x2 = ndi.gaussian_filter(disp[1], sigma=sigma)
    x3 = ndi.gaussian_filter(disp[2], sigma=sigma)
    return np.stack([x1, x2, x3], axis=0)


# from https://www.jianshu.com/p/8c9e9e57d48e
# def bilateralFilter(batch_img, ksize, sigmaColor=None, sigmaSpace=None, dim=3):
#     device = batch_img.device
#     if sigmaSpace is None:
#         sigmaSpace = 0.15 * ksize + 0.35  # 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
#     if sigmaColor is None:
#         sigmaColor = sigmaSpace

#     pad = (ksize - 1) // 2
#     batch_img_pad = F.pad(batch_img, pad=[pad]*(2*dim), mode='reflect')

#     # batch_img 的维度为 BxcxHxW, 因此要沿着第 二、三维度 unfold
#     # patches.shape:  B x C x H x W x ksize x ksize
#     patches = batch_img_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
#     patch_dim = patches.dim() # 6
#     # 求出像素亮度差
#     diff_intensity = patches - batch_img.unsqueeze(-1).unsqueeze(-1)
#     # 根据像素亮度差，计算权重矩阵
#     weights_color = torch.exp(-(diff_intensity ** 2) / (2 * sigmaColor ** 2))
#     # 归一化权重矩阵
#     weights_color = weights_color / weights_color.sum(dim=(-1, -2), keepdim=True)

#     # 获取 gaussian kernel 并将其复制成和 weight_color 形状相同的 tensor
#     weights_space = getGaussianKernel(ksize, sigmaSpace).to(device)
#     weights_space_dim = (patch_dim - 2) * (1,) + (ksize, ksize)
#     weights_space = weights_space.view(*weights_space_dim).expand_as(weights_color)

#     # 两个权重矩阵相乘得到总的权重矩阵
#     weights = weights_space * weights_color
#     # 总权重矩阵的归一化参数
#     weights_sum = weights.sum(dim=(-1, -2))
#     # 加权平均
#     weighted_pix = (weights * patches).sum(dim=(-1, -2)) / weights_sum
#     return weighted_pix


def anisotropic_filter(disp):
    # print("disp 0 ", disp[0].sum())
    x1 = anisodiff3(disp[0])
    # print("disp x1 ", x1.sum())
    x2 = anisodiff3(disp[1])
    x3 = anisodiff3(disp[2])
    return torch.stack([x1, x2, x3], axis=0)


def anisodiff3(
    stack, niter=4, kappa=50, gamma=0.1, step=(1, 1, 6.6), option=2, ploton=False
):
    # https://github.com/awangenh/fastaniso/blob/master/fastaniso.py
    """
    3D Anisotropic diffusion.
    Usage:
    stackout = anisodiff(stack, niter, kappa, gamma, option)
    Arguments:
            stack  - input stack
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (z,y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the middle z-plane will be plotted on every
                 iteration
    Returns:
            stackout   - diffused stack.
    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.
    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)
    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x,y and/or z axes
    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.
    Reference:
    P. Perona and J. Malik.
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    12(7):629-639, July 1990.
    Original MATLAB code by Peter Kovesi
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>
    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>
    June 2000  original version.
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    """

    # initialize output array
    stackout = stack.clone()

    # initialize some internal variables
    deltaS = torch.ones_like(stackout, device=stack.device)
    deltaE = deltaS.clone()
    deltaD = deltaS.clone()
    NS = deltaS.clone()
    EW = deltaS.clone()
    UD = deltaS.clone()
    gS = torch.ones_like(stackout, device=stack.device)
    gE = gS.clone()
    gD = gS.clone()

    for ii in range(niter):
        # calculate the diffs
        deltaD[:-1, :, :] = stackout[1:] - stackout[:-1]
        deltaS[:, :-1, :] = stackout[:, 1:] - stackout[:, :-1]
        deltaE[:, :, :-1] = stackout[:, :, 1:] - stackout[:, :, :-1]

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gD = torch.exp(-((deltaD / kappa) ** 2.0)) / step[0]
            gS = torch.exp(-((deltaS / kappa) ** 2.0)) / step[1]
            gE = torch.exp(-((deltaE / kappa) ** 2.0)) / step[2]
        elif option == 2:
            gD = 1.0 / (1.0 + (deltaD / kappa) ** 2.0) / step[0]
            gS = 1.0 / (1.0 + (deltaS / kappa) ** 2.0) / step[1]
            gE = 1.0 / (1.0 + (deltaE / kappa) ** 2.0) / step[2]

        # update matrices
        D = gD * deltaD
        E = gE * deltaE
        S = gS * deltaS

        # subtract a copy that has been shifted 'Up/North/West' by one
        # pixel. don't as questions. just do it. trust me.
        UD[:] = D
        NS[:] = S
        EW[:] = E
        UD[1:, :, :] -= D[:-1, :, :]
        NS[:, 1:, :] -= S[:, :-1, :]
        EW[:, :, 1:] -= E[:, :, :-1]

        # update the image
        stackout += gamma * (UD + NS + EW)

    return stackout


def filter_img_pair(img_tb1xyz_cpu):
    fix_img = img_tb1xyz_cpu[0]  # array of [b,1,x,y,z]
    mov_img = img_tb1xyz_cpu[-1]
    pre_batch = len(mov_img)
    for t in range(len(img_tb1xyz_cpu) - 1, -1, -1):
        cur_batch = len(img_tb1xyz_cpu[t])
        if cur_batch > pre_batch:
            mov_img = np.concatenate([mov_img, img_tb1xyz_cpu[t][pre_batch:]], axis=0)
        pre_batch = len(mov_img)
    return fix_img, mov_img


def pick_img_pair_for_pairwise(img_tb1xyz_cpu):
    fix_b1xyz = img_tb1xyz_cpu[0]
    mov_b1xyz = img_tb1xyz_cpu[1]
    for t in range(2, len(img_tb1xyz_cpu)):
        cur_batch = len(img_tb1xyz_cpu[t])
        mov_b1xyz = torch.cat([mov_b1xyz, img_tb1xyz_cpu[t]], dim=0)
        fix_b1xyz = torch.cat([fix_b1xyz, img_tb1xyz_cpu[0][:cur_batch]], dim=0)
    return fix_b1xyz, mov_b1xyz


def get_phase_array(root_path, phase_txt):
    phase_path = os.path.dirname((os.path.dirname(root_path)))
    phase_file = os.path.join(phase_path, phase_txt)
    phase_array = np.loadtxt(phase_file, delimiter=",")
    return phase_array


def pick_imgpair_first_mov_others_fix(img_tb1xyz_cpu):
    # print("len(img_tb1xyz_cpu): ", len(img_tb1xyz_cpu), img_tb1xyz_cpu[0].shape)
    fix_b1xyz = img_tb1xyz_cpu[1]
    mov_b1xyz = img_tb1xyz_cpu[0]
    for t in range(2, len(img_tb1xyz_cpu)):
        fix_b1xyz = torch.cat([fix_b1xyz, img_tb1xyz_cpu[t]], dim=0)
        mov_b1xyz = torch.cat([mov_b1xyz, img_tb1xyz_cpu[0]], dim=0)
    return fix_b1xyz, mov_b1xyz


def pick_imgpair_es_fixed(img_tb1xyz_cpu, es_idx):
    fix_b1xyz = img_tb1xyz_cpu[es_idx]
    mov_b1xyz = img_tb1xyz_cpu[0]
    for t in range(1, len(img_tb1xyz_cpu)):
        if t != es_idx:
            mov_b1xyz = torch.cat([mov_b1xyz, img_tb1xyz_cpu[t]], dim=0)
            fix_b1xyz = torch.cat([fix_b1xyz, img_tb1xyz_cpu[es_idx]], dim=0)
    return fix_b1xyz, mov_b1xyz


def print_list_shape(list_arr, name=" "):
    print(name, len(list_arr), end="  ")
    for i in range(len(list_arr)):
        print(list_arr[i].shape, end=", ")
    print("")


def change_all_tensors_to_numpy(tensor_list):
    def change_tensor_to_numpy(single_tensor):
        return single_tensor.cpu().numpy()

    return [change_tensor_to_numpy(single_tensor) for single_tensor in tensor_list]


def get_img_by_nibabel(image_path, device="cpu", get_spacing=False):
    img = nib.load(image_path)
    img_arr = img.get_fdata().astype("float32")  # [x,y,z,t]
    if get_spacing:
        img_spacing = list(img.header.get_zooms()[:3])
        img_spacing = [
            img_spacing[i].astype("float32") for i in range(len(img_spacing))
        ]
    img_arr = img_arr.transpose(3, 0, 1, 2)
    img_arr = torch.tensor(img_arr, device=device)
    if get_spacing:
        return img_arr, img_spacing
    else:
        return img_arr  # [t, x,y,z]


def sort_image_sequence(img_bt1xyz):
    batch_size = len(img_bt1xyz)
    img_bt1xyz = [img_bt1xyz[idx] for idx in range(batch_size)]
    seq_len_list = [img_bt1xyz[idx].shape[0] for idx in range(batch_size)]

    sorted_seq_len = sorted(
        enumerate(seq_len_list), key=lambda x: x[1], reverse=True
    )  # from max to min
    sorted_idx = [seq_len[0] for seq_len in sorted_seq_len]

    img_bt1xyz_sort = []
    for new_idx in sorted_idx:
        img_bt1xyz_sort.append(img_bt1xyz[new_idx])
    return img_bt1xyz_sort


def get_image_by_t(img_seq):
    seq_len_list = [len(img_seq[idx]) for idx in range(len(img_seq))]
    seq_len = max(seq_len_list)

    img_seq_by_t = []
    for t in range(seq_len):
        img_seq_t = get_image_t(img_seq, t, seq_len_list)  # [bs,1,x,y,z]
        img_seq_by_t.append(img_seq_t)
    return img_seq_by_t


def get_image_t(img_seq, t, seq_len_list):
    batch_size = len(img_seq)
    img_seq_t = []
    for idx in range(batch_size):
        if seq_len_list[idx] > t:
            img_seq_t.append(img_seq[idx][t])
        else:
            break
    img_seq_t = torch.stack(img_seq_t, dim=0)
    return img_seq_t


def integrate_displacement(full_disp, grid, n_steps):
    if n_steps > 0:
        full_disp = full_disp / (2**n_steps)
        for _ in range(n_steps):
            full_disp = full_disp + grid_sample_without_grid(full_disp, full_disp, grid)
    return full_disp


def KL_DIV(mu_q, sig_q, mu_p, sig_p):
    kl = (
        0.5
        * (
            2 * torch.log(sig_p / sig_q)
            - 1
            + (sig_q / sig_p).pow(2)
            + ((mu_p - mu_q) / sig_p).pow(2)
        ).mean()
    )
    return kl


def kld_gauss(mean_q, logvar_q):
    # 1-log(std^2)+mu^2+std^2, var=std^2=sigma^2
    kld_element = 1 - logvar_q + mean_q.pow(2) + torch.exp(logvar_q)
    return 0.5 * torch.mean(kld_element)


def kld_two_gauss(mu_p, logvar_p, mu_q, logvar_q):
    # KL(p||q)=1/2*(log(SIG_q/SIG_p)-k+(mu_p-mu_q)^2/SIG_q+SIG_p/SIG_q)
    # when q is default, mu_q=0 and logvar_q=0, SIG_q=1
    # KL(p||q) = 1/2*(-log(SIG_p)-k+(mu_p)^2+SIG_p), which is the formula in self._kld_gauss
    SIG_p = torch.exp(logvar_p)
    SIG_q = torch.exp(logvar_q)
    kld_element = (
        torch.log(SIG_q / SIG_p) - 1 + (mu_p - mu_q).pow(2) / SIG_q + (SIG_p / SIG_q)
    )
    return 0.5 * torch.mean(kld_element)


def reparameter(mean, logvar):
    eps = torch.FloatTensor(mean.size()).normal_(0, 1).to(mean.device)
    # eps = torch.autograd.Variable(eps)
    std = torch.exp(0.5 * logvar)
    return eps.mul(std).add_(mean)


def get_image_array_from_name(filename, orientation="xyz"):
    image_array = sitk.GetArrayFromImage(sitk.ReadImage(filename)).astype("float32")
    # SimpleITK读取的图像数据的坐标顺序为zyx，即从多少张切片到单张切片的宽和高；
    # 而据SimpleITK Image获取的origin和spacing的坐标顺序则是xyz
    # fid = open(filename.replace(".mhd", ".img"), 'rb')
    if orientation == "xyz":
        image_array = np.swapaxes(image_array, 0, 2)
    return image_array


def get_imgarr_by(dataset_name, case_id, phase_id, time_id, root_path):
    filename = dataset_name
    if dataset_name == "dirlab":
        filename += "_" + str(case_id) + "_" + str(phase_id)
        filename += ".nii.gz"
    elif dataset_name == "lung4d":
        filename += "_" + str(case_id) + "_" + str(time_id) + "_" + str(phase_id)
        filename += ".nii.gz"
    img_arr = get_image_array_from_name(os.path.join(root_path, filename))

    return img_arr


def jacobian_derterminant(disp):
    """
    https://itk.org/Doxygen/html/classitk_1_1DisplacementFieldJacobianDeterminantFilter.html
    :param phi: in the form of [batch, x,y,z,3]
    :return: [batch, x, y, z]
    """
    dx = torch_gradient(disp[..., 0])  # [batch, 3, x,y,z]
    dy = torch_gradient(disp[..., 1])
    dz = torch_gradient(disp[..., 2])
    detPhi = (
        (1 + dx[:, 0]) * (1 + dy[:, 1]) * (1 + dz[:, 2])
        + dx[:, 1] * dy[:, 2] * dz[:, 0]
        + dx[:, 2] * dy[:, 0] * dz[:, 1]
        - dx[:, 2] * (1 + dy[:, 1]) * dz[:, 0]
        - dx[:, 1] * dy[:, 0] * (1 + dz[:, 2])
        - (1 + dx[:, 0]) * dy[:, 2] * dz[:, 1]
    )
    return detPhi


def jacobian_derterminant2(disp):
    """
    https://itk.org/Doxygen/html/classitk_1_1DisplacementFieldJacobianDeterminantFilter.html
    :param phi: in the form of [batch, x,y,2]
    :return: [batch, x, y]
    """
    dx = torch_gradient2(disp[..., 0])  # [batch, 2, x,y]
    dy = torch_gradient2(disp[..., 1])
    detPhi = (1 + dx[:, 0]) * (1 + dy[:, 1]) - dx[:, 1] * dy[:, 0]
    return detPhi


def neg_Jac_percent(disp_channel_last, mask=None, return_num=False):
    """
    https://itk.org/Doxygen/html/classitk_1_1DisplacementFieldJacobianDeterminantFilter.html
    :param disp_channel_last: [batch, x,y,z,3]
    :param mask:
    :return:
    """
    jac_disp = jacobian_derterminant(disp_channel_last)  # [batch, x, y, z]
    img_size = list(jac_disp.shape[1:])
    num_pixels = np.prod(img_size)
    if mask is not None:
        mask[mask < 1] = 0
        num_pixels = mask.sum()
        if jac_disp.is_cuda:
            mask = torch.tensor(mask, device=jac_disp.device)
        jac_disp = jac_disp * mask

    num_neg_pixels = torch.sum(jac_disp <= 0)
    if num_neg_pixels.device is not None:
        num_neg_pixels = num_neg_pixels.cpu()
    num_neg_pixels = float(num_neg_pixels.numpy())
    neg_Jac_perc = round(100.0 * num_neg_pixels / num_pixels, 4)
    if return_num:
        return num_neg_pixels, neg_Jac_perc
    return neg_Jac_perc


def image_gradient(arr):
    """
    :param arr: [batch,c,x,y,z]
    :return: grad [batch, 3, x, y, z]
    """
    # 相反的顺序

    dim = len(arr.shape) - 2
    if dim == 3:
        arr = F.pad(arr, [0, 1, 0, 1, 0, 1], mode="replicate")

        gradx = arr[:, 0, 1:, :, :] - arr[:, 0, :-1, :, :]
        # grad_list[..., 0] = gradx[:,:, :-1, :-1,:]
        # grad_list.append(gradx[:,:, :-1, :-1,:])

        grady = arr[:, 0, :, 1:, :] - arr[:, 0, :, :-1, :]
        # grad_list[..., 1] = grady[:, :-1,:, :-1, :]
        # grad_list.append(grady[:,:-1, :, :-1,:])

        gradz = arr[:, 0, :, :, 1:] - arr[:, 0, :, :, :-1]
        # grad_list[..., 1] = gradz[:, :-1, :-1, :, :]
        # grad_list.append(gradz[:,:-1, :-1, :,:])

        grad = torch.stack(
            [gradx[:, :, :-1, :-1], grady[:, :-1, :, :-1], gradz[:, :-1, :-1, :]], dim=1
        )

    elif dim == 2:
        arr = F.pad(arr, [0, 1, 0, 1], mode="replicate")
        gradx = arr[:, 0, 1:, :] - arr[:, 0, :-1, :]
        grady = arr[:, 0, :, 1:] - arr[:, 0, :, :-1]

        grad = torch.stack([gradx[:, :, :-1], grady[:, :-1, :]], dim=1)

    return grad


def torch_gradient(arr):
    """
    :param arr: shape of [batch, x,y,z]
    :param spacing:
    :return: [batch, 3, x,y,z]
    """
    grad_list = []
    gradx = arr[:, 1:, :, :] - arr[:, :-1, :, :]
    grad_list.append(F.pad(gradx.unsqueeze(1), (0, 0, 0, 0, 0, 1), mode="replicate"))
    grady = arr[:, :, 1:, :] - arr[:, :, :-1, :]
    grad_list.append(F.pad(grady.unsqueeze(1), (0, 0, 0, 1, 0, 0), mode="replicate"))
    gradz = arr[:, :, :, 1:] - arr[:, :, :, :-1]
    grad_list.append(F.pad(gradz.unsqueeze(1), (0, 1, 0, 0, 0, 0), mode="replicate"))

    return torch.cat(grad_list, dim=1)


def torch_gradient2(arr):
    """
    :param arr: shape of [batch, x,y]
    :param spacing:
    :return: [batch, 2, x,y]
    """
    grad_list = []
    gradx = arr[:, 1:, :] - arr[:, :-1, :]
    grad_list.append(F.pad(gradx.unsqueeze(1), (0, 0, 0, 1), mode="replicate"))
    grady = arr[:, :, 1:] - arr[:, :, :-1]
    grad_list.append(F.pad(grady.unsqueeze(1), (0, 1, 0, 0), mode="replicate"))

    return torch.cat(grad_list, dim=1)


def normalize_gradient(gradient, eta=0.5):
    """

    :param gradient:[batch, 3, x,y,z]
    :param eta:
    :return:
    """
    tmp_gradient = 0
    dim = len(gradient.shape) - 2
    for i in range(dim):
        tmp_gradient += gradient[:, i].pow(2)
    tmp_gradient = tmp_gradient.unsqueeze(1)
    # epsilon = eta * torch.mean(torch.sqrt(tmp_gradient)).item()
    # print("tmp_gradient: ", tmp_gradient.shape)

    epsilon = 1e-5
    # print("norm: ", gradient.shape, ",tmp grad: ", tmp_gradient.shape)
    norm_grad = gradient / torch.sqrt(tmp_gradient + epsilon**2)
    return norm_grad


def normalize_image(image_array, min_thr, max_thr):
    image_array[image_array < min_thr] = min_thr
    image_array[image_array > max_thr] = max_thr
    image_array = (image_array - min_thr) / (max_thr - min_thr)

    return image_array


def write_array_to_image_file(
    image_array, filename, spacing, origin=(0, 0, 0), orientation="xyz"
):
    """

    :param image_array: [x,y,z] or [x,y,z,3] or [3, x, y, z], [1, x, y, z],  [x, y, z, 1]
    :param filename:
    :param spacing:
    :param origin:
    :param orientation:
    :return:
    """
    assert len(image_array.shape) == 3 or len(image_array.shape) == 4

    # [3, x, y, z] is processed into [x,y,z,3].
    # [1, x, y, z] and [x, y, z, 1] are processed into [x,y,z]
    if len(image_array.shape) == 4:
        if image_array.shape[0] == 3:
            image_array = np.transpose(image_array, (1, 2, 3, 0))
        elif image_array.shape[0] == 1 or image_array.shape[-1] == 1:
            image_array = image_array.squeeze()

    if orientation == "xyz":
        image_array = np.swapaxes(image_array, 0, 2)

    image_sitk = sitk.GetImageFromArray(image_array)
    image_sitk.SetSpacing(spacing)
    image_sitk.SetOrigin(origin)
    sitk.WriteImage(image_sitk, filename)
    return image_sitk


def create_batch_regular_grid(batch_size, img_size, device="cpu"):
    """

    :param batch_size:
    :param img_size:
    :param device:
    :return: channel last regular grid [batch, x,y,z,3]
    """
    dim = len(img_size)
    if dim == 3:
        D, H, W = img_size
        x_range = torch.tensor([i * 2 / (D - 1) - 1 for i in range(D)], device=device)
        y_range = torch.tensor([i * 2 / (H - 1) - 1 for i in range(H)], device=device)
        z_range = torch.tensor([i * 2 / (W - 1) - 1 for i in range(W)], device=device)

        regular_grid_list = torch.meshgrid(x_range, y_range, z_range, indexing="ij")
        regular_grid = torch.stack(regular_grid_list, dim=-1)
        batch_regular_grid = regular_grid.repeat(batch_size, 1, 1, 1, 1)

    elif dim == 2:
        D, H = img_size
        x_range = torch.tensor([i * 2 / (D - 1) - 1 for i in range(D)], device=device)
        y_range = torch.tensor([i * 2 / (H - 1) - 1 for i in range(H)], device=device)

        regular_grid_list = torch.meshgrid(x_range, y_range, indexing="ij")
        regular_grid = torch.stack(regular_grid_list, dim=-1)
        batch_regular_grid = regular_grid.repeat(batch_size, 1, 1, 1)

    return batch_regular_grid


def grid_sample_without_grid(
    inp,
    displacement,
    regular_grid=None,
    padding_mode="border",
    interp_mode="bilinear",
    align_corners=True,
):
    """
    no grid but flow
    :param inp: [batch, n, x, y, z]
    :param displacement: [batch, 3, x, y, z]
    :param regular_grid: [batch, x, y, z, 3]
    :param padding_mode:
    :param interp_mode:
    :return: [N,C,x,y,z]
    """
    shape = inp.shape[2:]
    dim = len(shape)
    device = displacement.device

    if regular_grid is None:
        batch_size = len(inp)
        regular_grid = create_batch_regular_grid(batch_size, shape, device)

    if dim == 3:
        disp_chan_last = displacement.permute(0, 2, 3, 4, 1).contiguous()
    elif dim == 2:
        disp_chan_last = displacement.permute(0, 2, 3, 1).contiguous()
    grid_channel_last = regular_grid + disp_chan_last

    output = grid_sample_with_grid(
        inp,
        grid_channel_last,
        padding_mode=padding_mode,
        interp_mode=interp_mode,
        align_corners=align_corners,
    )
    return output


def grid_sample_with_grid(
    inp,
    deformation_field,
    padding_mode="border",
    interp_mode="bilinear",
    align_corners=True,
):
    """
    :param inp: [batch, 1, x,y,z]
    :param deformation_field:  [batch, x,y,z, 3]
    :param padding_mode:
    :param interp_mode:
    :return:
    """
    grid_rev = torch.flip(deformation_field, [-1])  # flip the dim
    output_tensor = F.grid_sample(
        inp,
        grid_rev,
        padding_mode=padding_mode,
        mode=interp_mode,
        align_corners=align_corners,
    )
    # output_tensor is [N,C,x,y,z]
    return output_tensor


def lagrangian_flow(inf_flow, grid, forward=False):
    shape = inf_flow.shape  # flow is [t,3,x,y,z]
    seq_len = shape[0]
    lag_flow = torch.zeros(shape, device=inf_flow.device)
    lag_flow[0] = inf_flow[0]
    for k in range(1, seq_len):
        if forward:
            src = lag_flow[k]  # [3, x,y,z]
            sum_flow = inf_flow[k - 1 : k]
        else:
            src = inf_flow[k]
            sum_flow = lag_flow[k - 1 : k]
        src_x = src[0].unsqueeze(0).unsqueeze(0)  # [x,y,z] -> [1,1,x,y,z]
        src_y = src[1].unsqueeze(0).unsqueeze(0)
        src_z = src[2].unsqueeze(0).unsqueeze(0)
        lag_flow_x = grid_sample_with_grid(src_x, sum_flow)
        lag_flow_y = grid_sample_with_grid(src_y, sum_flow)
        lag_flow_z = grid_sample_with_grid(src_z, sum_flow)
        lag_flow[k] = sum_flow + torch.cat((lag_flow_x, lag_flow_y, lag_flow_z), dim=1)
    return lag_flow


def aug_affine(img, policies, policy_values):
    image_size = img.shape
    cuda_device = img.device
    pi = torch.asin(torch.tensor(1.0))

    grid = create_batch_regular_grid(1, image_size, cuda_device)[
        0
    ]  # grid is [1,x,y,z,3]==>[x,y,z,3]

    grid_4 = torch.cat(
        [grid, torch.ones(*[list(image_size) + [1]]).to(cuda_device)], dim=3
    )
    phi_x = torch.tensor(0.0).to(cuda_device)  # [-pi/6, pi/6] ==> [-30°, 30°]
    phi_y = torch.tensor(0.0).to(cuda_device)
    phi_z = torch.tensor(0.0).to(cuda_device)
    t_x = torch.tensor(0.0).to(cuda_device)
    t_y = torch.tensor(0.0).to(cuda_device)
    t_z = torch.tensor(0.0).to(cuda_device)
    # print("policies: %s, policy_values: %s"%(policies, policy_values))

    # rotate
    if "phi_x" in policies:
        phi_x = (
            ((policy_values["phi_x"] / 10).to(cuda_device) - 0.5) * pi / 6
        )  # [-pi/6, pi/6] ==> [-30°, 30°]
    if "phi_y" in policies:
        phi_x = ((policy_values["phi_y"] / 10).to(cuda_device) - 0.5) * pi / 6
    if "phi_z" in policies:
        phi_x = ((policy_values["phi_z"] / 10).to(cuda_device) - 0.5) * pi / 6
    if "t_x" in policies:
        t_x = ((policy_values["t_x"] / 10).to(cuda_device) - 0.5) * 10 / image_size[0]
    if "t_y" in policies:
        t_y = ((policy_values["t_y"] / 10).to(cuda_device) - 0.5) * 10 / image_size[1]
    if "t_z" in policies:
        t_z = ((policy_values["t_z"] / 10).to(cuda_device) - 0.5) * 10 / image_size[2]

    intensity_sum = torch.sum(img)
    center_mass_x = torch.sum(img * grid[..., 0]) / intensity_sum
    center_mass_y = torch.sum(img * grid[..., 1]) / intensity_sum
    center_mass_z = torch.sum(img * grid[..., 2]) / intensity_sum
    # print("phi: ", phi_x*180/pi, phi_y*180/pi,phi_z*180/pi)
    # print("phi: ", phi_x, phi_y,phi_z)
    # print("t: ", t_x, t_y, t_z)
    # print("center_mass: ", center_mass_x, center_mass_y, center_mass_z)

    trans_matrix_pos = torch.diag(torch.ones(4)).to(cuda_device)
    trans_matrix_cm = torch.diag(torch.ones(4)).to(cuda_device)
    trans_matrix_cm_rw = torch.diag(torch.ones(4)).to(cuda_device)

    trans_matrix_pos[0, 3] = t_x
    trans_matrix_pos[1, 3] = t_y
    trans_matrix_pos[2, 3] = t_z

    trans_matrix_cm[0, 3] = -center_mass_x
    trans_matrix_cm[1, 3] = -center_mass_y
    trans_matrix_cm[2, 3] = -center_mass_z

    trans_matrix_cm_rw[0, 3] = center_mass_x
    trans_matrix_cm_rw[1, 3] = center_mass_y
    trans_matrix_cm_rw[2, 3] = center_mass_z

    R_x = torch.diag(torch.ones(4)).to(cuda_device)
    R_x[1, 1] = torch.cos(phi_x)
    R_x[1, 2] = -torch.sin(phi_x)
    R_x[2, 1] = torch.sin(phi_x)
    R_x[2, 2] = torch.cos(phi_x)

    R_y = torch.diag(torch.ones(4)).to(cuda_device)
    R_y[0, 0] = torch.cos(phi_y)
    R_y[0, 2] = torch.sin(phi_y)
    R_y[2, 0] = -torch.sin(phi_y)
    R_y[2, 2] = torch.cos(phi_y)

    R_z = torch.diag(torch.ones(4)).to(cuda_device)
    R_z[0, 0] = torch.cos(phi_z)
    R_z[0, 1] = -torch.sin(phi_z)
    R_z[1, 0] = torch.sin(phi_z)
    R_z[1, 1] = torch.cos(phi_z)
    rotation_matrix = torch.mm(torch.mm(R_z, R_y), R_x)
    transformation_matrix = torch.mm(
        torch.mm(torch.mm(trans_matrix_pos, trans_matrix_cm), rotation_matrix),
        trans_matrix_cm_rw,
    )[0:3, :]
    dense_displacement = (
        torch.mm(
            grid_4.view(np.prod(image_size).tolist(), 4).contiguous(),
            transformation_matrix.t(),
        )
        .view(*(image_size), 3)
        .contiguous()
        - grid_4[..., :3]
    )

    new_img = grid_sample_with_grid(
        img.unsqueeze(0).unsqueeze(0), (dense_displacement + grid).unsqueeze(0)
    )

    return new_img


def affine(img, seed_id=42):
    """

    :param img: shape of [x,y,z], must be tensor
    :return: shape of [x,y,z]
    """
    # t0 = time.time()
    image_size = img.shape
    cuda_device = img.device
    pi = torch.asin(torch.tensor(1.0))

    grid = create_batch_regular_grid(1, image_size, cuda_device)[
        0
    ]  # grid is [1,x,y,z,3]==>[x,y,z,3]

    grid_4 = torch.cat(
        [grid, torch.ones(*[list(image_size) + [1]]).to(cuda_device)], dim=3
    )
    torch.manual_seed(seed_id)
    phi_x = (
        (torch.rand(1).to(cuda_device) - 0.5) * pi / 9
    )  # [-pi/9, pi/9] ==> [-20°, 20°]
    torch.manual_seed(seed_id + 1)
    phi_y = (torch.rand(1).to(cuda_device) - 0.5) * pi / 9
    torch.manual_seed(seed_id + 2)
    phi_z = (torch.rand(1).to(cuda_device) - 0.5) * pi / 9
    torch.manual_seed(seed_id + 3)
    t_x = (torch.rand(1).to(cuda_device) - 0.5) * 10 / image_size[0]
    torch.manual_seed(seed_id + 4)
    t_y = (torch.rand(1).to(cuda_device) - 0.5) * 10 / image_size[1]
    torch.manual_seed(seed_id + 5)
    t_z = (torch.rand(1).to(cuda_device) - 0.5) * 10 / image_size[2]

    intensity_sum = torch.sum(img)
    center_mass_x = torch.sum(img * grid[..., 0]) / intensity_sum
    center_mass_y = torch.sum(img * grid[..., 1]) / intensity_sum
    center_mass_z = torch.sum(img * grid[..., 2]) / intensity_sum
    # print("phi: ", phi_x*180/pi, phi_y*180/pi,phi_z*180/pi)
    # print("phi: ", phi_x, phi_y,phi_z)
    # print("t: ", t_x, t_y, t_z)
    # print("center_mass: ", center_mass_x, center_mass_y, center_mass_z)

    trans_matrix_pos = torch.diag(torch.ones(4)).to(cuda_device)
    trans_matrix_cm = torch.diag(torch.ones(4)).to(cuda_device)
    trans_matrix_cm_rw = torch.diag(torch.ones(4)).to(cuda_device)

    trans_matrix_pos[0, 3] = t_x
    trans_matrix_pos[1, 3] = t_y
    trans_matrix_pos[2, 3] = t_z

    trans_matrix_cm[0, 3] = -center_mass_x
    trans_matrix_cm[1, 3] = -center_mass_y
    trans_matrix_cm[2, 3] = -center_mass_z

    trans_matrix_cm_rw[0, 3] = center_mass_x
    trans_matrix_cm_rw[1, 3] = center_mass_y
    trans_matrix_cm_rw[2, 3] = center_mass_z

    R_x = torch.diag(torch.ones(4)).to(cuda_device)
    R_x[1, 1] = torch.cos(phi_x)
    R_x[1, 2] = -torch.sin(phi_x)
    R_x[2, 1] = torch.sin(phi_x)
    R_x[2, 2] = torch.cos(phi_x)

    R_y = torch.diag(torch.ones(4)).to(cuda_device)
    R_y[0, 0] = torch.cos(phi_y)
    R_y[0, 2] = torch.sin(phi_y)
    R_y[2, 0] = -torch.sin(phi_y)
    R_y[2, 2] = torch.cos(phi_y)

    R_z = torch.diag(torch.ones(4)).to(cuda_device)
    R_z[0, 0] = torch.cos(phi_z)
    R_z[0, 1] = -torch.sin(phi_z)
    R_z[1, 0] = torch.sin(phi_z)
    R_z[1, 1] = torch.cos(phi_z)
    rotation_matrix = torch.mm(torch.mm(R_z, R_y), R_x)
    transformation_matrix = torch.mm(
        torch.mm(torch.mm(trans_matrix_pos, trans_matrix_cm), rotation_matrix),
        trans_matrix_cm_rw,
    )[0:3, :]
    dense_displacement = (
        torch.mm(
            grid_4.view(np.prod(image_size).tolist(), 4).contiguous(),
            transformation_matrix.t(),
        )
        .view(*(image_size), 3)
        .contiguous()
        - grid_4[..., :3]
    )

    new_img = grid_sample_with_grid(
        img.unsqueeze(0).unsqueeze(0), (dense_displacement + grid).unsqueeze(0)
    )
    # print("cost ", time.time()-t0, cuda_device)
    return new_img[0][0]


def plot_lines(
    axes, flow_slice, point, margin, sp, is_horizontal, is_vertical, regular_grid
):
    slice_shape = flow_slice[0].shape
    zero_min = max(point[0] - margin, 0)
    zero_max = min(point[0] + margin, slice_shape[0] - 1)
    first_min = max(point[1] - margin, 0)
    first_max = min(point[1] + margin, slice_shape[1] - 1)

    # 先画横线
    if is_horizontal:
        for kth in range(first_min, first_max, sp):
            axes.plot(
                [
                    temp + flow_slice[0][temp, kth]
                    for temp in range(zero_min, zero_max, sp)
                ],
                kth + flow_slice[1][zero_min:zero_max:sp, kth],
                color="r",
                linewidth=0.5,
            )
            if regular_grid:
                axes.plot(
                    [temp for temp in range(zero_min, zero_max, sp)],
                    [kth] * len(range(zero_min, zero_max, sp)),
                    color="black",
                    linewidth=0.2,
                )

    # 再画纵线
    if is_vertical:
        for kth in range(zero_min, zero_max, sp):
            axes.plot(
                kth + flow_slice[0][kth, first_min:first_max:sp],
                [
                    temp + flow_slice[1][kth, temp]
                    for temp in range(first_min, first_max, sp)
                ],
                color="r",
                linewidth=0.5,
            )
            if regular_grid:
                axes.plot(
                    [kth] * len(range(first_min, first_max, sp)),
                    [temp for temp in range(first_min, first_max, sp)],
                    color="black",
                    linewidth=0.2,
                )
    plt.axis([0, slice_shape[0] - 1, 0, slice_shape[1] - 1])


def plot_grid(
    img3d_list,
    flow_list,
    center_point,
    margin=10,
    sp=5,
    show_image=True,
    regular_grid=False,
    is_vertical=True,
    is_horizontal=True,
    root_path=None,
):
    """

    :param img3d_list: list of images,
        In the order of [fixed, warped_fcn, warped_seq, moving], image in the shape of [x,y,z]
    :param flow_list: list of flows
        In the order of [flow_fcn, flow_seq], flow in the shape of [x,y,z,3]
    :param center_point: list of shape 3, [int, int, int]
        The coordinate of the center point that cuts.
    :param margin: int
        the margin of the presented grid to the center_point.
    :param sp: int
        The spacing of grid
    :param show_image:
    :param regular_grid:
    :param is_vertical:
    :param is_horizontal:
    :return:
    """
    # prepare data
    # view_names = ["transverse plane", "coronal plane", "sagittal plane"]
    # col_names = ["Fixed", "FCN-based grid", "Proposed grid", "Moving"]
    x_pos, y_pos, z_pos = center_point
    # the center point in certain cross section
    point2d_list = [[x_pos, y_pos], [x_pos, z_pos], [y_pos, z_pos]]

    if show_image:
        img2d_list = []
        # img2d_list includes different slices
        # img3d[:, :, z_pos] is to cut the img3d using a plane perpendicular to the Z axis, the cut position is z_pos.
        # ::-1 in the last axis is due to the direction of the image
        for col, img3d in enumerate(img3d_list):
            img2d_list.append(
                [img3d[:, :, z_pos], img3d[:, y_pos, :], img3d[x_pos, :, :]]
            )

    # flow2d_list stores the flow slice to plot grid
    flow2d_list = []
    for flow in flow_list:
        # print("flow shape ", flow.shape)
        flow2d_z = [flow[:, :, z_pos, i] for i in [0, 1]]
        flow2d_y = [flow[:, y_pos, :, i] for i in [0, 2]]
        flow2d_x = [flow[x_pos, :, :, i] for i in [1, 2]]  # [2][y,z,]

        flow2d_list.append([flow2d_z, flow2d_y, flow2d_x])

    # start plot
    fig = plt.figure(dpi=300)
    nrows = 3
    ncols = 4
    height_ratios = [1, 1, 1]
    width_ratios = [1, 1, 1, 1]
    wspace = 0.025
    hspace = 0.025
    spec = gridspec.GridSpec(
        ncols=ncols,
        nrows=nrows,
        wspace=wspace,
        hspace=hspace,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
    )

    row_title = ["a", "b", "c"]
    col_title = ["1", "2", "3", "4"]
    for col in range(ncols):
        for row in range(nrows):
            axes = fig.add_subplot(spec[row * ncols + col])
            # for the first row, add title
            # if row == 0:
            #     plt.title(col_names[col], fontsize='small')

            if show_image:
                plt.imshow(
                    img2d_list[col][row].T, cmap="gray", origin="lower", aspect="auto"
                )

            if col == 1 or col == 2:
                # the ith in flow2d list, the jth slice,
                slice = flow2d_list[col - 1][row]  # [2][y,z]
                point = point2d_list[row]
                plot_lines(
                    axes,
                    slice,
                    point,
                    margin,
                    sp,
                    is_horizontal,
                    is_vertical,
                    regular_grid,
                )

            if row == 0:
                axes.invert_yaxis()  # 反转Y坐标轴

                r, c = img2d_list[col][row].shape
                plt.text(
                    r * 0.01,
                    c * 0.1,
                    "(" + row_title[row] + col_title[col] + ")",
                    fontsize="small",
                    fontweight="roman",
                    color="blue",
                )
            else:
                r, c = img2d_list[col][row].shape
                plt.text(
                    r * 0.01,
                    c * 0.9,
                    "(" + row_title[row] + col_title[col] + ")",
                    fontsize="small",
                    fontweight="roman",
                    color="blue",
                )

            plt.xticks([])
            plt.yticks([])
            # plt.axis('off')
            # print("rwo %d, column %d, done\n"%(row, col))
    # plt.tight_layout()
    plt.subplots_adjust(
        wspace=wspace, hspace=hspace, left=0.03, right=0.97, bottom=0.03, top=0.97
    )

    plt.savefig(
        os.path.join(
            root_path,
            "disp_sp%d_image%d_regular%d_vertical%d_horizontal%d.pdf"
            % (sp, show_image, regular_grid, is_vertical, is_horizontal),
        )
    )


def apply_rand_transform(image_tensor):
    torch.random.manual_seed(42)
    disp_arr = torch.rand([1, 3, 8, 8, 7], device=image_tensor.device)
    # disp_arr[:, :2] = disp_arr[:, :2] / 64 * 1.85
    # disp_arr[:, 2] = disp_arr[:, 2] / 56 * 1.15
    disp_arr[:, :2] = disp_arr[:, :2] / 64 * 1.90
    disp_arr[:, 2] = disp_arr[:, 2] / 56 * 1.0
    disp_arr = F.upsample(disp_arr, scale_factor=8, mode="trilinear")
    warped_image_tensor = grid_sample_without_grid(
        image_tensor.unsqueeze(0).unsqueeze(0), disp_arr, interp_mode="nearest"
    ).squeeze()
    return warped_image_tensor


def apply_rand_transform_with_mask(image_tensor, mask):
    # image_tensor is (x,y,z), mask is (x, y, z)
    torch.random.manual_seed(42)
    disp_arr = (torch.rand([1, 3, 8, 8, 7], device=image_tensor.device) - 0.5) * 2
    # disp_arr[:, :2] = disp_arr[:, :2] / 64 * 1.85
    # disp_arr[:, 2] = disp_arr[:, 2] / 56 * 1.15
    # mask = F.upsample(
    #     mask.float().unsqueeze(0).unsqueeze(0),
    #     scale_factor=1 / 8,
    #     mode="nearest",
    # )
    disp_arr = disp_arr
    # 这是对于mov=ED的情况。
    # disp_arr[:, :2] = disp_arr[:, :2] / 64 * 4.0
    # disp_arr[:, 2] = disp_arr[:, 2] / 56 * 1.5

    # 对于mov=ES的情况，要增加形变，使得结果好看一点。。
    disp_arr[:, :2] = disp_arr[:, :2] / 64 * 6.0
    disp_arr[:, 2] = disp_arr[:, 2] / 56 * 1.5

    mask_arr = (1 - mask.float()).unsqueeze(0).unsqueeze(0)
    disp_arr = F.upsample(disp_arr, scale_factor=8, mode="trilinear") * mask_arr
    warped_image_tensor = grid_sample_without_grid(
        image_tensor.unsqueeze(0).unsqueeze(0), disp_arr, interp_mode="nearest"
    ).squeeze()
    return warped_image_tensor  # , mask.float().squeeze().numpy()
