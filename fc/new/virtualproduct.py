import xarray as xr
from itertools import product

from datacube.virtual import Transformation, Measurement

FC_MEASUREMENTS = [
    {
        'name': 'PV',
        'dtype': 'int8',
        'nodata': -1,
        'units': 'percent'
    },
    {
        'name': 'NPV',
        'dtype': 'int8',
        'nodata': -1,
        'units': 'percent'
    },
    {
        'name': 'BS',
        'dtype': 'int8',
        'nodata': -1,
        'units': 'percent'
    },
    {
        'name': 'UE',
        'dtype': 'int8',
        'nodata': -1,
        'units': ''
    },
]


class FractionalCover(Transformation):
    """ Applies the fractional cover algorithm to surface reflectance data.
    Requires bands named 'green', 'red', 'nir', 'swir1', 'swir2'
    """

    def __init__(self, *args, **kwargs):
        self.output_measurements = {m['name']: Measurement(**m) for m in FC_MEASUREMENTS}

    def measurements(self, input_measurements):
        return self.output_measurements

    def compute(self, data):
        pass

    def compute_multidim(self, data):
        from fc.fractional_cover import fractional_cover
        sel = [dict(p)
               for p in product(*[[(i.name, i.item()) for i in c]
                                  for v, c in data.coords.items()
                                  if v not in data.geobox.dims])]
        fc = []
        for s in sel:
            fc.append(fractional_cover(data.sel(**s), self.output_measurements))
        return xr.concat(fc, dim='time')


class FakeFractionalCover(Transformation):
    """ Applies the fractional cover algorithm to surface reflectance data.
    Requires bands named 'green', 'red', 'nir', 'swir1', 'swir2'
    """

    def __init__(self, *args, **kwargs):
        self.output_measurements = {m['name']: Measurement(**m) for m in FC_MEASUREMENTS}

    def measurements(self, input_measurements):
        return self.output_measurements

    def compute(self, data):
        return xr.Dataset({'BS': data.red, 'PV': data.green, 'NPV': data.nir, 'UE': data.swir1},
                          attrs=data.attrs)
