from __future__ import print_function
import numpy as np
from torchvision import transforms as T


def generate_data(M, N, d, wavelength, SNR, doa_min, NUM_REPEAT, grid, GRID_NUM):
    data = []
    label = []

    for doa_idx in range(GRID_NUM):
        DOA = doa_min + grid * doa_idx

        for rep_idx in range(NUM_REPEAT):
            add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
            array_signal = 0

            signal_i = 10 ** (SNR / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
            phase_shift_unit = 2 * np.pi * d / wavelength * np.sin(DOA / 180 * np.pi)
            a_i_ = np.cos(np.array(range(M)) * phase_shift_unit) + 1j * np.sin(np.array(range(M)) * phase_shift_unit)
            a_i = np.expand_dims(a_i_, axis=-1)
            array_signal_i = np.matmul(a_i, signal_i)
            array_signal += array_signal_i

            array_output_nf = array_signal + 0 * add_noise  # noise-free output
            array_output = array_signal + 1 * add_noise

            array_covariance_nf = 1 / N * (np.matmul(array_output_nf, np.matrix.getH(array_output_nf)))
            array_covariance = 1 / N * (np.matmul(array_output, np.matrix.getH(array_output)))

            aa = np.array(array_covariance.real)
            # aa=T.ToTensor()
            bb = np.array(array_covariance.imag)
            # bb = T.ToTensor()
            cc=np.array([aa,bb])
            #cc= cc.transpose()
            #dd=Permute((10,10,2))(cc)

            data.append(cc)

            aa_nf=np.asarray(array_covariance_nf.real)
            # aa_nf = T.ToTensor()
            bb_nf=np.asarray(array_covariance_nf.imag)
            # bb_nf = T.ToTensor()
            cc_nf=np.array([aa_nf,bb_nf])
            label.append(cc_nf)
    data = np.asarray(data)
    label = np.asarray(label)
    return data,label

def generate_spec_batches(data, batch_size):

    data_ = data['input']
    label_ = data['label']
    data_len = len(label_)

    # shuffle data
    shuffle_seq = np.random.permutation(range(data_len))
    data = [data_[idx] for idx in shuffle_seq]
    label = [label_[idx] for idx in shuffle_seq]

    # generate batches   生成批次
    num_batch = int(data_len / batch_size)  # 3
    data_batches = []
    label_batches = []
    for batch_idx in range(num_batch):
        batch_start = batch_idx * batch_size
        batch_end = np.min([(batch_idx + 1) * batch_size, data_len])
        data_batch = data[batch_start : batch_end]
        label_batch = label[batch_start: batch_end]
        data_batches.append(data_batch)
        label_batches.append(label_batch)

    return data_batches, label_batches

