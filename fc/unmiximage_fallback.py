import scipy.optimize
import numpy


def unmiximage(weighted_spectra, endmembers_array, in_null, out_unmix_null):
    output_terms = len(endmembers_array[0]) + 1
    image_shape = (output_terms,) + weighted_spectra.shape[1:]
    fractions = numpy.empty(image_shape)
    it = numpy.nditer(fractions[0], flags=['multi_index'])
    while not it.finished:
        index = (slice(None),) + it.multi_index
        solution_index = (slice(0, -1),) + it.multi_index
        err_index = (slice(-1, None),) + it.multi_index
        solution, err = scipy.optimize.nnls(endmembers_array, weighted_spectra[index])
        fractions[solution_index] = solution
        fractions[err_index] = err
        it.iternext()
    return fractions
