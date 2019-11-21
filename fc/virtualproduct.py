import xarray as xr
from itertools import product

from datacube.virtual import Transformation, Measurement
from fc import __version__
from fc.fractional_cover import fractional_cover

FC_MEASUREMENTS = [
    {
        'name': 'pv',
        'dtype': 'int8',
        'nodata': -1,
        'units': 'percent'
    },
    {
        'name': 'npv',
        'dtype': 'int8',
        'nodata': -1,
        'units': 'percent'
    },
    {
        'name': 'bs',
        'dtype': 'int8',
        'nodata': -1,
        'units': 'percent'
    },
    {
        'name': 'ue',
        'dtype': 'int8',
        'nodata': -1,
        'units': ''
    },
]


class FractionalCover(Transformation):
    """
    Applies the fractional cover algorithm to surface reflectance data.
    Requires bands named 'green', 'red', 'nir', 'swir1', 'swir2'
    """

    def __init__(self, regression_coefficients=None, c2_scaling=False):
        if regression_coefficients is None:
            regression_coefficients = {band: [0, 1]
                                       for band in ['green', 'red', 'nir', 'swir1', 'swir2']
                                       }
        self.regression_coefficients = regression_coefficients
        self.c2_scaling = c2_scaling

    def measurements(self, input_measurements):
        return {m['name']: Measurement(**m) for m in FC_MEASUREMENTS}

    def compute(self, data):
        # Typically creates a list of dictionaries looking like [{time: 1234}, {time: 1235}, ...]
        if self.c2_scaling:
            # The C2 data need to be scaled
            scale_usgs_collection2(data)

        sel = [dict(p)
               for p in product(*[[(i.name, i.item()) for i in c]
                                  for v, c in data.coords.items()
                                  if v not in data.geobox.dims])]
        fc = []
        measurements = [Measurement(**m) for m in FC_MEASUREMENTS]
        for s in sel:
            fc.append(fractional_cover(data.sel(**s), measurements, self.regression_coefficients))
        fc = xr.concat(fc, dim='time')
        fc.attrs['crs'] = data.attrs['crs']
        try:
            fc = fc.rename(BS='bs', PV='pv', NPV='npv', UE='ue')
        except ValueError:  # Assuming the names are already correct and don't need to be changed.
            pass
        return fc

    def algorithm_metadata(self):
        return {
            'algorithm': {
                'name': 'Fractional Cover',
                'version': __version__,
                'repo_url': 'https://github.com/GeoscienceAustralia/fc.git',
                'parameters': {
                    'regression_coefficients': self.regression_coefficients,
                    'usgs_c2_scaling': self.c2_scaling
                }
            }}


class FakeFractionalCover(FractionalCover):
    """
    Fake (fast) fractional cover for testing purposes only

    Requires bands named 'green', 'red', 'nir', 'swir1', 'swir2'
    """

    def compute(self, data):
        if self.c2_scaling:
            # The C2 data need to be scaled
            scale_usgs_collection2(data)
        return xr.Dataset({'blue': data.blue,
                           'red': data.red,
                           'green': data.green},
                          attrs=data.attrs)


def scale_usgs_collection2(data):
    return data.map(scale_and_clip_dataarray, keep_attrs=True,
                    scale_factor=2.75, add_offset=-2000, clip_range=(0, 10000))


def scale_and_clip_dataarray(dataarray: xr.DataArray, *, scale_factor=1, add_offset=0, clip_range=None):
    nodata = dataarray.attrs['nodata']
    mask = dataarray.data == nodata
    dataarray.data = dataarray.data * scale_factor + add_offset
    dataarray.data[mask] = nodata
    if clip_range is not None:
        clip_min, clip_max = clip_range
        dataarray.clip(clip_min, clip_max)
    return dataarray
