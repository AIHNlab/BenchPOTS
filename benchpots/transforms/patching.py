"""
Contextual patching functions for time series data.
<#TODO: insert publication info> 
"""

# Created by Rafael Morand <rafael.morand@unibe.ch>
# License: BSD-3-Clause

from typing import Union

import numpy as np
import torch

from ..utils.logging import logger


def patched_timeseries(time_series: Union[np.ndarray, torch.Tensor],
                       main_freq: int, 
                       n_subsequences_per_patch: int, 
                       len_subsequence: int, 
                       stride: int, 
                       repeat_subsequences: bool=False, 
                       return_indeces: bool=False, 
                       return_as_sorted_img: bool=False, 
                       inplace: bool=False):
    ''' Make patches from time series data according to a main frequency. The main frequency is determined by a recurring
    event in the data (e.g., the daily cycle in weather data, the circadian rhythm in activity tracker data, etc.). 
    The patches are made by sampling subsequences of length len_subsequence at multiples of the main frequency.
    The patched time series can be returned as a time series with the corresponding indeces from the original series, or
    as a sorted image, where the subsequences are sorted by feature.
    
    Parameters
    ----------
    time_series :
        time series data, len(shape)=2, [total_length, feature_num]
        
    main_freq :
        The main frequency of the time series. The subsequences starts at multiples of this frequency.
        
    len_subsequence :
        The length of each subsequence in the patch.
        
    stride :
        The stride between the start point of the patches.
        
    n_subsequences_per_patch :
        The number of subsequences in the patch. (if the main frequency is the daily cycle, then
        n_subsequences_per_patch=7 equals a weekly patch)
        
    repeat_subsequences :
        If True, the patches are shifted by 1 stride each time. 
        If False, the patches are primarily shifted by 1 stride for main_freq/stride times, before aplying a secondary
        shift of main_freq*n:subsequences_per_patch.
        It is recommended to set to True for forecasting tasks and False for imputation tasks.
        
    inplace :
        If True, the function will modify the input time_series. If False, the function will return a new array.
        
    Returns
    -------
    patches (depends on condition) :
        If return_as_sorted_img=False: The patches of shape [patch_num, n_subsequences_per_patch * len_subsequence, feature_num].
        If return_as_sorted_img=True: The patches of shape [patch_num, n_subsequences_per_patch * feature_num, len_subsequence].
        
    indeces (returned if return_indeces=True) :
        The indeces of the samples for the patches as they appear in the full time series.
        
    '''
    time_series, indeces = patcher(time_series,
                                   main_freq, 
                                   len_subsequence, 
                                   stride, 
                                   n_subsequences_per_patch, 
                                   repeat_subsequences=repeat_subsequences, 
                                   inplace=inplace)
    
    # reshape to time series [patch_num, n_subsequences_per_patch * len_subsequence, feature_num]
    time_series = time_series.reshape(time_series.shape[0], -1, time_series.shape[-1]) 
    
    if return_as_sorted_img:
        if return_indeces:
            logger.info("indeces are not returned if return_as_sorted_img==True.")
        return timeseries_to_sorted_image(time_series, len_subsequence)
    
    if not return_indeces:
        return time_series
    else:
        indeces = indeces.reshape(indeces.shape[0], -1)
        return time_series, indeces


def patcher(time_series: Union[np.ndarray, torch.Tensor],
            main_freq: int, 
            len_subsequence: int, 
            stride: int, 
            n_subsequences_per_patch: int, 
            repeat_subsequences: bool=False, 
            inplace: bool=False):
    ''' Make patches from time series data according to a main frequency. The main frequency is determined by a recurring
    event in the data (e.g., the daily cycle in weather data, the circadian rhythm in activity tracker data, etc.). 
    The patches are made by sampling subsequences of length len_subsequence at multiples of the main frequency.
    
    Parameters
    ----------
    time_series :
        time series data, len(shape)=2, [total_length, feature_num]
        
    main_freq :
        The main frequency of the time series. The subsequences starts at multiples of this frequency.
        
    len_subsequence :
        The length of each subsequence in the patch.
        
    stride :
        The stride between the start point of the patches.
        
    n_subsequences_per_patch :
        The number of subsequences in the patch. (if the main frequency is the daily cycle, then
        n_subsequences_per_patch=7 equals a weekly patch)
        
    repeat_subsequences :
        If True, the patches are shifted by 1 stride each time. 
        If False, the patches are primarily shifted by 1 stride for main_freq/stride times, before aplying a secondary
        shift of main_freq*n:subsequences_per_patch.
        It is recommended to set to True for forecasting tasks and False for imputation tasks.
        
    inplace :
        If True, the function will modify the input time_series. If False, the function will return a new array.
        
    Returns
    -------
    patches :
        The patches of shape [patch_num, n_subsequences_per_patch, len_subsequence, feature_num].
        
    indeces :
        The indeces of the samples for the patches as they appear in the full time series.
        
    '''
    __assert_input(
        time_series=time_series, 
        main_freq=main_freq, 
        len_subsequence=len_subsequence, 
        stride=stride, 
        n_subsequences_per_patch=n_subsequences_per_patch
    )
    
    if not inplace:
        time_series = time_series.copy()
        
    time_series = extend_sequence_length_for_patching(time_series, main_freq, n_subsequences_per_patch)
    
    # make patches
    patches, indeces = [], []
    if repeat_subsequences:
        patch_num = determine_number_of_patches(time_series, main_freq, stride, n_subsequences_per_patch)
        for k in range(patch_num):
            offset = k*stride
            patch, idxs = make_single_patch(time_series, main_freq, len_subsequence, n_subsequences_per_patch, offset)
            patches.append(patch), indeces.append(idxs)
    else:
        patch_num_per_subseq = main_freq // stride
        n_rows = time_series.shape[0] // (main_freq*n_subsequences_per_patch)
        for k in range(n_rows):
            base_offset = k * main_freq * n_subsequences_per_patch
            for r in range(patch_num_per_subseq):
                offset = r * stride + base_offset
                patch, idxs = make_single_patch(time_series, main_freq, len_subsequence, n_subsequences_per_patch, offset)
                patches.append(patch), indeces.append(idxs)
    
    # return depending on type
    if isinstance(time_series, torch.Tensor):
        return torch.stack(patches), torch.stack(indeces)
    elif isinstance(time_series, np.ndarray):
        return np.array(patches), np.array(indeces)
    else:
        raise RuntimeError(f"Input time_series should be either torch.Tensor or np.ndarray. Got {type(time_series)}.")


def determine_number_of_patches(time_series: Union[np.ndarray, torch.Tensor],
                                main_freq: int, 
                                stride: int, 
                                n_subsequences_per_patch: int):
    ''' Determine the number of patches that can be sampled from the time series. The number of patches N is given by:
    
    N = (T - (F * (M - 1))) / S, with
    
    T: the total length of the time series, 
    
    F: the main frequency, 
    
    M: the number of subsequences per patch
    
    S: the stride

    Parameters
    ----------
    time_series :
        time series data, len(shape)=2, [total_length, feature_num]
        
    main_freq :
        The main frequency of the time series. The subsequences starts at multiples of this frequency.
        
    stride :
        The stride between the start point of the patches.
        
    n_subsequences_per_patch :
        The number of subsequences in the patch.

    Returns
    -------
    patch_num :
        The number of unique patches that can be sampled from the given time series.
    
    '''
    patch_num = (time_series.shape[0] - (main_freq * (n_subsequences_per_patch - 1))) / stride
    patch_num = int(np.ceil(patch_num)) 
    return patch_num
        
        
def make_single_patch(time_series: Union[np.ndarray, torch.Tensor],
                      main_freq: int, 
                      len_subsequence: int,
                      n_subsequences_per_patch: int, 
                      offset: int):      
    ''' Sample a patch from the time series. Each subsequence of the patch starts at a multiple of the main frequency 
    and has length len_subsequence. 

    Parameters
    ----------
    time_series :
        time series data, len(shape)=2, [total_length, feature_num]
        
    main_freq :
        The main frequency of the time series. The subsequences starts at multiples of this frequency.
        
    len_subsequence :
        The length of each subsequence in the patch.
        
    n_subsequences_per_patch :
        The number of subsequences in the patch.
        
    offset :
        The offset  of the first subsequence in the patch relative to the full time series.

    Returns
    -------
    patch :
        The generated patch of shape [n_subsequences_per_patch, len_subsequence, feature_num].
        
    indeces :
        The indeces of the samples for the patch as they appear in the full time series.

    ''' 
    indeces = []
    for i in range(n_subsequences_per_patch):
        t0 = i * main_freq + offset
        t1 = t0 + len_subsequence
        indeces.append(np.arange(t0, t1))
    return time_series[indeces], indeces


def extend_sequence_length_for_patching(time_series: Union[np.ndarray, torch.Tensor],
                                        main_freq: int, 
                                        n_subsequences_per_patch: int):
    ''' Extends the sequence length to be a multiple of the main frequency. Adds nan values to the end of the sequence. 

    Parameters
    ----------
    time_series :
        time series data, len(shape)=2, [total_length, feature_num]
        
    main_freq :
        The main frequency of the time series. The subsequences starts at multiples of this frequency.
        
    n_subsequences_per_patch :
        The number of subsequences in the patch.

    Returns
    -------
    time_series :
        The extended time series with nan values in the end.

    '''
    n_to_add = (n_subsequences_per_patch * main_freq) - (time_series.shape[0] % (n_subsequences_per_patch * main_freq)) 
    if isinstance(time_series, torch.Tensor):
        time_series = torch.cat([time_series, torch.full((n_to_add, time_series.shape[-1]), np.nan)])
    elif isinstance(time_series, np.ndarray):
        time_series = np.concatenate([time_series, np.full((n_to_add, time_series.shape[-1]), np.nan)])
    else:
        raise RuntimeError(f"Input time_series should be either torch.Tensor or np.ndarray. Got {type(time_series)}.")
    return time_series


def timeseries_to_sorted_image(time_series: Union[np.ndarray, torch.Tensor],
                               len_subsequence: int):
    ''' Converts a time series to a "sorted" image of shape [patch_num, n_subsequences_per_patch * feature_num, len_subsequence]. 
    The term "sorted" refers to the fact that the subsequences are sorted by features. I.e., the first n_subsequences 
    are the first feature, the next n_subsequences are the second feature, etc.
    
    Parameters
    ----------
    time_series :
        time series data, len(shape)=2, [total_length, feature_num]
        
    len_subsequence :
        The length of each subsequence in the patch.
        
    Returns
    -------
    time_series :
        The time series reshaped to an image of shape [patch_num, n_subsequences_per_patch * feature_num, len_subsequence]
        
    '''
    time_series = time_series.reshape(time_series.shape[0], -1, len_subsequence, time_series.shape[-1])
    
    if isinstance(time_series, torch.Tensor):
        time_series = time_series.permute(0, 3, 1, 2)
    elif isinstance(time_series, np.ndarray):
        time_series = time_series.transpose(0, 3, 1, 2)
    else:
        raise RuntimeError(f"Input time_series should be either torch.Tensor or np.ndarray. Got {type(time_series)}.")
    
    time_series = time_series.reshape(time_series.shape[0], -1, time_series.shape[-1])
    return time_series


def __assert_input(time_series: Union[np.ndarray, torch.Tensor],
                   main_freq: int, 
                   n_subsequences_per_patch: int,  
                   stride: int,
                   len_subsequence: int):
    ''' Asserts the input for the patching functions (dimensions, dtypes, values). Returns nothing if all checks pass. '''
    assert np.ndim(time_series) == 2, f"Input time_series should be 2D [total_length, feature_num], but got {time_series.shape}."
    assert type(main_freq) == int, f"main_freq should be an integer, but got {type(main_freq)}."
    assert type(n_subsequences_per_patch) == int, (
        f"n_subsequences_per_patch should be an integer, but got {type(n_subsequences_per_patch)}." )
    assert type(len_subsequence) == int, f"len_subsequence should be an integer, but got {type(len_subsequence)}."
    assert type(stride) == int, f"stride should be an integer, but got {type(stride)}."
    assert (main_freq > 0 and len_subsequence > 0 and stride > 0 and n_subsequences_per_patch > 0), (
        "Inputs {main_freq, len_subsequence, stride, n_subsequences_per_patch} should be positive.")
    assert stride <= len_subsequence, (
        f"stride {stride} shouldn't be larger than sequence length len_subsequence {len_subsequence}. "
        f"Otherwise there will be gaps between samples."
    )