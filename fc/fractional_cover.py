from __future__ import absolute_import

import numpy
import numexpr
import xarray
from functools import partial
try:
    import dask.array
    dask_array_type = (dask.array.Array,)
except ImportError:  # pragma: no cover
    dask_array_type = ()

from datacube import Datacube
from datacube.storage.masking import valid_data_mask
from datacube.utils import iter_slices

from . import endmembers
from .unmix import unmiximage


DEFAULT_MEASUREMENTS = [{
    'name': 'PV',
    'dtype': 'int8',
    'nodata': -1,
    'units': 'percent'
}]


def fractional_cover(nbar_tile, measurements=None, regression_coefficients=None):
    """
    Given a tile of spectral observations compute the fractional components.
    The data should be a 2D array

    :param xarray.Dataset nbar_tile:
        A dataset with the following data variables (0-10000):
            * green
            * red
            * nir
            * swir1
            * swir2

    :param list(dict) measurements:
        A list of measurement item dicts, each containing:
            * name - name of output data_var
            * src_var - (optional) if `name` is not one of `['PV', 'NPV', 'BS', 'UE']`, use one of them here
            * dtype - dtype to use, eg `'int8'`
            * nodata - value to fill in for no data, eg `-1`
            * units' - eg `'percent'`

    :param dict regression_coefficients:
        A dictionary with six pairs of coefficients to apply to the green, red, nir, swir1 and swir2 values
        (blue is not used)

    :return:
        An xarray.Dataset containing:
            * Green vegetation (PV)
            * Non-green vegetation (NPV)
            * Bare soil (BS)
            * Unmixing error (UE)

    :rtype:
        xarray.Dataset
    """
    if measurements is None:
        measurements = DEFAULT_MEASUREMENTS

    # Ensure the bands are all there and in the right order
    nbar_tile = nbar_tile[['green', 'red', 'nir', 'swir1', 'swir2']]

    # Set nodata to 0
    no_data = 0
    is_valid_array = valid_data_mask(nbar_tile).to_array(dim='band').all(dim='band')

    nbar = nbar_tile.to_array(dim='band')
    nbar = nbar.where(is_valid_array, no_data)

    output_data = compute_fractions(nbar.data, regression_coefficients)

    def data_func(measurement):
        band_names = ['PV', 'NPV', 'BS', 'UE']
        src_var = measurement.get('src_var', None) or measurement.get('name')
        i = band_names.index(src_var)

        # Set nodata value into output array
        band_nodata = numpy.dtype(measurement['dtype']).type(measurement['nodata'])
        unmasked_var = output_data[i]
        no_error = (unmasked_var != -1)

        where = numpy.where if not isinstance(unmasked_var, dask_array_type) else dask.array.where
        return where(is_valid_array.data & no_error, unmasked_var, band_nodata)

    dataset = Datacube.create_storage({}, nbar_tile.geobox, measurements, data_func)

    return dataset


def make_temp_array(nbar):
    geo_shape = nbar.shape[1:]
    temp_vars = ['green', 'dead1', 'dead2', 'bare', 'err']
    temp_shape = (len(temp_vars),) + geo_shape
    return numpy.empty(temp_shape, dtype=numpy.float)


#: pylint: disable=too-many-locals
def compute_fractions(nbar, regression_coefficients):
    """
    Compute the fractional cover of the given imagery tile

    :param numpy.array nbar: Input array of [green, red, nir, swir1, swir2] * (x, y)
    :return (numpy.array, numpy_array): Output array of [green, dead, bare] * (x, y), and the unmix error array
    """
    if isinstance(nbar, dask_array_type):
        lazy_compute_fractions = partial(_compute_fractions, regression_coefficients=regression_coefficients)
        new_chunks = ((4,),) + nbar.chunks[1:]
        out = dask.array.map_blocks(lazy_compute_fractions, nbar.rechunk({0: -1}), chunks=new_chunks, dtype='int8')
        return out
    else:
        return _compute_fractions(nbar, regression_coefficients)


#: pylint: disable=too-many-locals
def _compute_fractions(nbar, regression_coefficients):
    temp_arr = make_temp_array(nbar)

    sum_to_one_weight = endmembers.sum_weight()
    endmembers_array = endmembers.get_endmembers(sum_to_one_weight)

    band_index = (slice(0, None),)

    # calculate in chunks to stay under 2GB mem limit
    chunk_size = (50, 4000)
    for geo_index in iter_slices(nbar.shape[1:], chunk_size):
        index = band_index + geo_index
        arr = nbar[index]
        temp_arr[index] = unmix(arr[0], arr[1], arr[2], arr[3], arr[4], sum_to_one_weight, endmembers_array,
                                regression_coefficients)

    green, dead1, dead2, bare, err = temp_arr

    # Find unmixing errors - if an pixel is in error then all pixels for that location are errors
    wh_unmix_err = numexpr.evaluate("(green == -10) |"
                                    "(dead1 == -10) |"
                                    "(dead2 == -10) |"
                                    "(bare == -10)")

    # scale the results and clip the range to (0, 100)
    green = numexpr.evaluate("green / 0.01")
    numpy.clip(green, a_min=0, a_max=127, out=green)

    dead = numexpr.evaluate("(dead1 + dead2) / 0.01")
    numpy.clip(dead, a_min=0, a_max=127, out=dead)

    bare = numexpr.evaluate("bare / 0.01")
    numpy.clip(bare, a_min=0, a_max=127, out=bare)

    err = numexpr.evaluate("err")
    numpy.clip(err, a_min=0, a_max=127, out=err)

    output_data = numpy.array([green, dead, bare, err], dtype=numpy.int8)
    output_data[:, wh_unmix_err] = -1

    return output_data


def apply_coefficients_for_band(numpyarray, band, regression_coefficients):
    """
    Apply regression coefficients in the form: ETM = c0 + OLI*c1
    As per table 2 in http://www.mdpi.com/2072-4292/6/9/7952/htm
    :param numpyarray: array of measurements to apply coefficients to
    :param band: name of the coefficient pair to apply
    :return: updated array
    """
    band_coefficients = regression_coefficients[band]
    coefficient0 = band_coefficients[0]
    coefficient1 = band_coefficients[1]
    numpyarray = numexpr.evaluate('coefficient0 + (coefficient1 * numpyarray)')
    return numpyarray


#: pylint: disable=too-many-statements, too-many-locals
def unmix(green, red, nir, swir1, swir2, sum_to_one_weight, endmembers_array, regression_coefficients):
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
    :param numpy.Array swir2:
    :param float sum_to_one_weight: Scale factor
    :param numpy.Array endmembers_array: Endmembers array
    """

    if regression_coefficients is not None:
        green = apply_coefficients_for_band(green, 'green', regression_coefficients)
        red = apply_coefficients_for_band(red, 'red', regression_coefficients)
        nir = apply_coefficients_for_band(nir, 'nir', regression_coefficients)
        swir1 = apply_coefficients_for_band(swir1, 'swir1', regression_coefficients)
        swir2 = apply_coefficients_for_band(swir2, 'swir2', regression_coefficients)

    band2 = numexpr.evaluate("(1.0 + green) * 0.0001")
    band3 = numexpr.evaluate("(1.0 + red) * 0.0001")
    band4 = numexpr.evaluate("(1.0 + nir) * 0.0001")
    band5 = numexpr.evaluate("(1.0 + swir1) * 0.0001")
    band7 = numexpr.evaluate("(1.0 + swir2) * 0.0001")

    # b_logs = numexpr.evaluate("log(subset)")
    logb2 = numexpr.evaluate("log(band2)")
    logb3 = numexpr.evaluate("log(band3)")
    logb4 = numexpr.evaluate("log(band4)")
    logb5 = numexpr.evaluate("log(band5)")
    logb7 = numexpr.evaluate("log(band7)")

    b2b3 = numexpr.evaluate("band2 * band3")
    b2b4 = numexpr.evaluate("band2 * band4")
    b2b5 = numexpr.evaluate("band2 * band5")
    b2b7 = numexpr.evaluate("band2 * band7")
    b2lb2 = numexpr.evaluate("band2 * logb2")
    b2lb3 = numexpr.evaluate("band2 * logb3")
    b2lb4 = numexpr.evaluate("band2 * logb4")
    b2lb5 = numexpr.evaluate("band2 * logb5")
    b2lb7 = numexpr.evaluate("band2 * logb7")

    b3b4 = numexpr.evaluate("band3 * band4")
    b3b5 = numexpr.evaluate("band3 * band5")
    b3b7 = numexpr.evaluate("band3 * band7")
    b3lb2 = numexpr.evaluate("band3 * logb2")
    b3lb3 = numexpr.evaluate("band3 * logb3")
    b3lb4 = numexpr.evaluate("band3 * logb4")
    b3lb5 = numexpr.evaluate("band3 * logb5")
    b3lb7 = numexpr.evaluate("band3 * logb7")

    b4b5 = numexpr.evaluate("band4 * band5")
    b4b7 = numexpr.evaluate("band4 * band7")
    b4lb2 = numexpr.evaluate("band4 * logb2")
    b4lb3 = numexpr.evaluate("band4 * logb3")
    b4lb4 = numexpr.evaluate("band4 * logb4")
    b4lb5 = numexpr.evaluate("band4 * logb5")
    b4lb7 = numexpr.evaluate("band4 * logb7")

    b5b7 = numexpr.evaluate("band5 * band7")
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

    # 2014_07_23 uses 60 endmembers
    interactive_terms = numpy.array([b2b3, b2b4, b2b5, b2b7, b2lb2, b2lb3, b2lb4, b2lb5, b2lb7,
                                     b3b4, b3b5, b3b7, b3lb2, b3lb3, b3lb4, b3lb5, b3lb7,
                                     b4b5, b4b7, b4lb2, b4lb3, b4lb4, b4lb5, b4lb7,
                                     b5b7, b5lb2, b5lb3, b5lb4, b5lb5, b5lb7,
                                     b7lb2, b7lb3, b7lb4, b7lb5, b7lb7, lb2lb3,
                                     lb2lb4, lb2lb5, lb2lb7, lb3lb4, lb3lb5, lb3lb7, lb4lb5, lb4lb7, lb5lb7,
                                     band2, band3, band4, band5, band7,
                                     logb2, logb3, logb4, logb5, logb7,
                                     band_ratio1, band_ratio2, band_ratio3, band_ratio4])

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

    # gives green, dead1, dead2 and bare fractions
    # the last band should be the unmixing error
    return fractions
