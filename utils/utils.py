import numpy as np
import tensorflow as tf
PRECISION = 'float32'
EPS = 1E-8

# Data I/O
def load_nii(filename):
    import nibabel as nib
    return nib.load(filename).get_data()

def save_nii(data, filename, filename_sample=''):
    import nibabel as nib
    if filename_sample:
        nib.save(nib.Nifti1Image(data, None, nib.load(filename_sample).header), filename)
    else:
        nib.save(nib.Nifti1Image(data, None, None), filename)

def load_h5(filename, varname='data'):
    import h5py
    with h5py.File(filename, 'r') as f:
        data = f[varname][:]
    return data

def save_h5(data, filename, varname='data'):
    import h5py
    with h5py.File(filename, 'w') as f:
        f.create_dataset(varname, data=data)
        
def load_mat(filename, varname='data'):
    try:
        import scipy.io as sio
        f = sio.loadmat(filename)
        data = f[varname]        
    except:
        data = load_h5(filename, varname=varname)
        if data.ndim == 4:
            data = data.transpose(3,2,1,0)
        elif data.ndim == 3:
            data = data.transpose(2,1,0)
    return data
        
def load_dicom(foldername, flag_info=True):
    import pydicom
    import os
    foldername, _, filenames = next(os.walk(foldername))
    filenames = sorted(filenames)
    data, info = [], {}
    slice_min, loc_min, slice_max, loc_max = None, None, None, None
    for filename in filenames:
        dataset = pydicom.dcmread('{0}/{1}'.format(foldername, filename))
        data.append(dataset.pixel_array)
        # Voxel size
        info['voxel_size'] = tuple(map(float, list(dataset.PixelSpacing) + [dataset.SpacingBetweenSlices]))
        # Slice location
        if slice_min is None or slice_min > float(dataset.SliceLocation):
            slice_min = float(dataset.SliceLocation)
            loc_min = np.array(dataset.ImagePositionPatient)
        if slice_max is None or slice_max < float(dataset.SliceLocation):
            slice_max = float(dataset.SliceLocation)
            loc_max = np.array(dataset.ImagePositionPatient)
    data = np.stack(data, axis=-1)
    # Matrix size
    info['matrix_size'] = data.shape
    # B0 direction
    affine2D = np.array(dataset.ImageOrientationPatient).reshape(2,3).T
    affine3D = (loc_max - loc_min) / ((info['matrix_size'][2]-1)*info['voxel_size'][2])
    affine3D = np.concatenate((affine2D, affine3D.reshape(3,1)), axis=1)
    info['B0_dir'] = tuple(np.dot(np.linalg.inv(affine3D), np.array([0, 0, 1])))
    if flag_info:
        return data, info
    else:
        return data

def save_dicom(data, foldername_tgt, foldername_src):
    import pydicom
    import os
    if not os.path.exists(foldername_tgt):
        os.mkdir(foldername_tgt)
    foldername_src, _, filenames = next(os.walk(foldername_src))
    filenames = sorted(filenames)
    for i, filename in enumerate(filenames):
        dataset = pydicom.dcmread('{0}/{1}'.format(foldername_src, filename))
        dataset.PixelData = data[..., i].tobytes()
        dataset.save_as('{0}/{1}'.format(foldername_tgt, filename))

# dipole kernel in Fourier space
def dipole_kernel(matrix_size, voxel_size, B0_dir):
    
    x = np.arange(-matrix_size[1]/2, matrix_size[1]/2, 1)
    y = np.arange(-matrix_size[0]/2, matrix_size[0]/2, 1)
    z = np.arange(-matrix_size[2]/2, matrix_size[2]/2, 1)
    Y, X, Z = np.meshgrid(x, y, z)
    
    X = X/(matrix_size[0]*voxel_size[0])
    Y = Y/(matrix_size[1]*voxel_size[1])
    Z = Z/(matrix_size[2]*voxel_size[2])
    
    D = 1/3 - (X*B0_dir[0] + Y*B0_dir[1] + Z*B0_dir[2])**2/(X**2 + Y**2 + Z**2)
    D[np.isnan(D)] = 0
    D = np.fft.fftshift(D);
    return D

def dataterm_mask(N_std, Mask, Normalize=True):
    w = Mask/N_std
    w[np.isnan(w)] = 0
    w[np.isinf(w)] = 0
    w = w*(Mask>0)
    if Normalize:
        w = w/np.mean(w[Mask>0])     
    return w
