from __future__ import absolute_import

import numpy
import numexpr
from math import ceil

import xarray

from datacube.storage.masking import valid_data_mask

from . import endmembers
from .unmix import unmiximage


def endmembers_version():
    return '2014_07_23'


def fractional_cover(nbar_tile):
    """
    Given a tile of spectral observations compute the fractional components.
    The data should be a 2D array

    :param xarray.Dataset nbar_tile:
        A dataset with the following data variables (0-10000:
            * green
            * red
            * nir
            * swir1
            * swir2

    :return:
        An xarray.Dataset containing:
            * Green vegetation (PV)
            * Non-green vegetation (NPV)
            * Bare soil (BS)
            * Unmixing error (UE)

    :rtype:
        xarray.Dataset
    """

    # Set nodata to 0
    no_data = 0
    is_valid_array = valid_data_mask(nbar_tile).to_array(dim='band').all(dim='band')

    nbar = nbar_tile.to_array(dim='band')
    nbar.data[:, ~is_valid_array.data] = no_data

    output_data, output_err = compute_fractions(nbar.data)

    output_data[:, ~is_valid_array.data] = no_data
    output_err[~is_valid_array.data] = no_data

    dims = ('band',) + nbar.dims[1:]
    coords = {k: v for k, v in nbar.coords.items()}
    coords['band'] = ['PV', 'NPV', 'BS']
    dataset = xarray.DataArray(output_data, dims=dims, coords=coords).to_dataset(dim='band')
    dataset.attrs = nbar_tile.attrs
    dataset['UE'] = (dims[1:], output_err)

    return dataset


def compute_fractions(nbar):
    """
    Compute the fractional cover of the given imagery tile

    :param numpy.array nbar: Input array of [green, red, nir, swir1, swir2] * (x, y)
    :return (numpy.array, numpy_array): Output array of [green, dead, bare] * (x, y), and the unmix error array
    """
    geo_shape = nbar.shape[1:]

    temp_vars = ['green', 'dead1', 'dead2', 'bare', 'err']
    temp_shape = (len(temp_vars),) + geo_shape
    temp_arr = numpy.empty(temp_shape, dtype=numpy.float)

    sum_to_one_weight = endmembers.sum_weight(endmembers_version())
    endmembers_array = endmembers.get_endmembers(endmembers_version(), sum_to_one_weight)

    band_index = (slice(0, None),)

    # calculate in chunks to stay under 2GB mem limit
    chunk_size = (75, 4000)
    for geo_index in iter_slices(geo_shape, chunk_size):
        index = band_index + geo_index
        arr = nbar[index]
        temp_arr[index] = unmix(arr[0], arr[1], arr[2], arr[3], arr[4], sum_to_one_weight, endmembers_array)

    green, dead1, dead2, bare, err = temp_arr

    # Find unmixing errors - if an pixel is in error then all pixels for that location are errors
    wh_unmix_err = numexpr.evaluate("(green == -10) |"
                                    "(dead1 == -10) |"
                                    "(dead2 == -10) |"
                                    "(bare == -10)")

    # scale the results and clip the range to (0, 255)
    green = numexpr.evaluate("green / 0.01 + 100")
    numpy.clip(green, a_min=0, a_max=255, out=green)

    dead = numexpr.evaluate("(dead1 + dead2) / 0.01 + 100")
    numpy.clip(dead, a_min=0, a_max=255, out=dead)

    bare = numexpr.evaluate("bare / 0.01 + 100")
    numpy.clip(bare, a_min=0, a_max=255, out=bare)

    output_data = numpy.array([green, dead, bare], dtype=numpy.uint8)
    output_data[:, wh_unmix_err] = 0

    err[wh_unmix_err] = 0

    return output_data, err


def unmix(green, red, nir, swir1, swir2, sum_to_one_weight, endmembers_array):
    """
    NNLS Unmixing v1.0
    Scarth 20090810 14:06:35 CEST
    This implements a constrained unmixing process to recover the fraction images from
    a synthetic reflectance generated from a large number of interactive
    terms produced from the original and log-transformed landsat bands

    GA wrapped and modified version of Scarth 20090810 14:06:35 CEST

    :param numpy.Array green:
    :param numpy.Array red:
    :param numpy.Array nir:
    :param numpy.Array swir1:
    :param numpy.Array swir1:
    :param float sum_to_one_weight: Scale factor
    :param numpy.Array endmembers_array: Endmembers array
    """

    band2 = numexpr.evaluate("(1.0 + green) * 0.0001")
    band3 = numexpr.evaluate("(1.0 + red) * 0.0001")
    band4 = numexpr.evaluate("(1.0 + nir) * 0.0001")
    band5 = numexpr.evaluate("(1.0 + swir1) * 0.0001")
    band7 = numexpr.evaluate("(1.0 + swir2) * 0.0001")

    #b_logs = numexpr.evaluate("log(subset)")
    logb2 = numexpr.evaluate("log(band2)")
    logb3 = numexpr.evaluate("log(band3)")
    logb4 = numexpr.evaluate("log(band4)")
    logb5 = numexpr.evaluate("log(band5)")
    logb7 = numexpr.evaluate("log(band7)")

    b2b3  = numexpr.evaluate("band2 * band3")
    b2b4  = numexpr.evaluate("band2 * band4")
    b2b5  = numexpr.evaluate("band2 * band5")
    b2b7  = numexpr.evaluate("band2 * band7")
    b2lb2 = numexpr.evaluate("band2 * logb2")
    b2lb3 = numexpr.evaluate("band2 * logb3")
    b2lb4 = numexpr.evaluate("band2 * logb4")
    b2lb5 = numexpr.evaluate("band2 * logb5")
    b2lb7 = numexpr.evaluate("band2 * logb7")

    b3b4  = numexpr.evaluate("band3 * band4")
    b3b5  = numexpr.evaluate("band3 * band5")
    b3b7  = numexpr.evaluate("band3 * band7")
    b3lb2 = numexpr.evaluate("band3 * logb2")
    b3lb3 = numexpr.evaluate("band3 * logb3")
    b3lb4 = numexpr.evaluate("band3 * logb4")
    b3lb5 = numexpr.evaluate("band3 * logb5")
    b3lb7 = numexpr.evaluate("band3 * logb7")

    b4b5  = numexpr.evaluate("band4 * band5")
    b4b7  = numexpr.evaluate("band4 * band7")
    b4lb2 = numexpr.evaluate("band4 * logb2")
    b4lb3 = numexpr.evaluate("band4 * logb3")
    b4lb4 = numexpr.evaluate("band4 * logb4")
    b4lb5 = numexpr.evaluate("band4 * logb5")
    b4lb7 = numexpr.evaluate("band4 * logb7")

    b5b7  = numexpr.evaluate("band5 * band7")
    b5lb2 = numexpr.evaluate("band5 * logb2")
    b5lb3 = numexpr.evaluate("band5 * logb3")
    b5lb4 = numexpr.evaluate("band5 * logb4")
    b5lb5 = numexpr.evaluate("band5 * logb5")
    b5lb7 = numexpr.evaluate("band5 * logb7")

    b7lb2 = numexpr.evaluate("band7 * logb2")
    b7lb3 = numexpr.evaluate("band7 * logb3")
    b7lb4 = numexpr.evaluate("band7 * logb4")
    b7lb5 = numexpr.evaluate("band7 * logb5")
    b7lb7 = numexpr.evaluate("band7 * logb7")

    lb2lb3 = numexpr.evaluate("logb2 * logb3")
    lb2lb4 = numexpr.evaluate("logb2 * logb4")
    lb2lb5 = numexpr.evaluate("logb2 * logb5")
    lb2lb7 = numexpr.evaluate("logb2 * logb7")

    lb3lb4 = numexpr.evaluate("logb3 * logb4")
    lb3lb5 = numexpr.evaluate("logb3 * logb5")
    lb3lb7 = numexpr.evaluate("logb3 * logb7")

    lb4lb5 = numexpr.evaluate("logb4 * logb5")
    lb4lb7 = numexpr.evaluate("logb4 * logb7")

    lb5lb7 = numexpr.evaluate("logb5 * logb7")

    band_ratio1 = numexpr.evaluate("(band4 - band3) / (band4 + band3)")
    band_ratio2 = numexpr.evaluate("(band4 - band5) / (band4 + band5)")
    band_ratio3 = numexpr.evaluate("(band5 - band3) / (band5 + band3)")
    band_ratio4 = numexpr.evaluate("(band3 - band2) / (band3 + band2)")

    # The 2009_08_10 and 2012_12_07 versions use a different interactive
    # terms array compared to the 2013_01_08 version
    # 2013_01_08 uses 59 endmebers
    # 2009_08_10 uses 56 endmebers
    # 2012_12_07 uses 56 endmebers
    # 2014_07_23 uses 60 endmembers
    # TODO write an interface that can retrieve the correct
    # interactiveTerms array according to the specified version.

    interactive_terms = numpy.array([b2b3, b2b4, b2b5, b2b7, b2lb2, b2lb3,
                                     b2lb4, b2lb5, b2lb7, b3b4, b3b5, b3b7,
                                     b3lb2, b3lb3, b3lb4, b3lb5, b3lb7, b4b5,
                                     b4b7, b4lb2, b4lb3, b4lb4, b4lb5, b4lb7,
                                     b5b7, b5lb2, b5lb3, b5lb4, b5lb5, b5lb7,
                                     b7lb2, b7lb3, b7lb4, b7lb5, b7lb7, lb2lb3,
                                     lb2lb4, lb2lb5, lb2lb7, lb3lb4, lb3lb5,
                                     lb3lb7, lb4lb5, lb4lb7, lb5lb7, band2,
                                     band3, band4, band5, band7, logb2, logb3,
                                     logb4, logb5, logb7, band_ratio1,
                                     band_ratio2, band_ratio3, band_ratio4])

    # Now add the sum to one constraint to the interactive terms
    # First make a zero array of the right shape
    weighted_spectra = numpy.zeros((interactive_terms.shape[0] + 1,) +
                                   interactive_terms.shape[1:])
    # Insert the interactive terms
    weighted_spectra[:-1, ...] = interactive_terms
    # Last element is special weighting
    weighted_spectra[-1] = sum_to_one_weight

    in_null = 0.0001
    out_unmix_null = -10.0

    fractions = unmiximage.unmiximage(weighted_spectra, endmembers_array, in_null, out_unmix_null)

    # 2013v gives green, dead1, dead2 and bare fractions
    # the last band should be the unmixing error
    return fractions


def iter_slices(shape, chunk_size):
    """
    Generates slices for a given shape

    E.g. ``shape=(4000, 4000), chunk_size=(500, 500)``
    Would yield 64 tuples of slices, each indexing 500x500.

    If the shape is not divisible by the chunk_size, the last chunk in each dimension will be smaller.

    :param tuple(int) shape: Shape of an array
    :param tuple(int) chunk_size: length of each slice for each dimension
    :return: Yeilds slices that can be used on an array of the given shape
    """
    assert len(shape) == len(chunk_size)
    num_grid_chunks = [int(ceil(s/float(c))) for s, c in zip(shape, chunk_size)]
    for grid_index in numpy.ndindex(*num_grid_chunks):
        slices = tuple(slice(min(d*c, stop), min((d+1)*c, stop))
                  for d, c, stop in zip(grid_index, chunk_size, shape))
        yield slices
