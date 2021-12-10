import xarray as xr
from datacube.virtual import Measurement, Transformation

from fc import __version__
from fc.fractional_cover import fractional_cover

FC_MEASUREMENTS = [
    {"name": "pv", "dtype": "int8", "nodata": -1, "units": "percent"},
    {"name": "npv", "dtype": "int8", "nodata": -1, "units": "percent"},
    {"name": "bs", "dtype": "int8", "nodata": -1, "units": "percent"},
    {"name": "ue", "dtype": "int8", "nodata": -1, "units": ""},
]


class FractionalCover(Transformation):
    """
    Applies the fractional cover algorithm to surface reflectance data.
    Requires bands named 'green', 'red', 'nir', 'swir1', 'swir2'
    """

    def __init__(
        self,
        regression_coefficients=None,
        c2_scaling=False,
        test_mode=False,
        clip_after_regression=False,
    ):
        self.regression_coefficients = regression_coefficients
        self.c2_scaling = c2_scaling
        self.test_mode = test_mode
        self.clip_after_regression = clip_after_regression

    @property
    def measurements(self):
        return [Measurement(**m) for m in FC_MEASUREMENTS]

    def compute(self, data):
        if self.test_mode:
            # Downsample to a size which will run quickly
            data = data.isel(x=slice(0, 100), y=slice(0, 100))
        if self.c2_scaling:
            # The C2 data need to be scaled
            data = scale_usgs_collection2(data)

        fc = []

        for time_idx in range(len(data.time)):
            fc.append(
                fractional_cover(
                    data.isel(time=time_idx),
                    self.measurements,
                    self.regression_coefficients,
                    clip_after_regression=self.clip_after_regression
                )
            )
        fc = xr.concat(fc, dim="time")
        fc.attrs["crs"] = data.attrs["crs"]
        try:
            fc = fc.rename(BS="bs", PV="pv", NPV="npv", UE="ue")
        except ValueError:  # Assuming the names are already correct and don't need to be changed.
            pass
        return fc

    def algorithm_metadata(self):
        return {
            "algorithm": {
                "name": "Fractional Cover",
                "version": __version__,
                "repo_url": "https://github.com/GeoscienceAustralia/fc.git",
                "parameters": {
                    "regression_coefficients": self.regression_coefficients,
                    "usgs_c2_scaling": self.c2_scaling,
                },
            }
        }


class FakeFractionalCover(FractionalCover):
    """
    Fake (fast) fractional cover for testing purposes only

    Requires bands named 'green', 'red', 'nir', 'swir1', 'swir2'
    """

    def compute(self, data):
        if self.c2_scaling:
            # The C2 data need to be scaled
            data = scale_usgs_collection2(data)
        return xr.Dataset(
            {"pv": data.red, "bs": data.red, "npv": data.green}, attrs=data.attrs
        )


def scale_usgs_collection2(data):
    return data.apply(
        scale_and_clip_dataarray,
        keep_attrs=True,
        scale_factor=0.275,
        add_offset=-2000,
        clip_range=(0, 10000),
    )


def scale_and_clip_dataarray(
    dataarray: xr.DataArray,
    scale_factor=1,
    add_offset=0,
    clip_range=None,
    new_nodata=-999,
    new_dtype="int16",
):
    orig_attrs = dataarray.attrs
    nodata = dataarray.attrs["nodata"]

    mask = dataarray.data == nodata

    dataarray = dataarray * scale_factor + add_offset

    if clip_range is not None:
        dataarray = dataarray.clip(*clip_range)

    dataarray = dataarray.astype(new_dtype)

    dataarray.data[mask] = new_nodata

    dataarray.attrs = orig_attrs
    dataarray.attrs["nodata"] = new_nodata

    return dataarray
