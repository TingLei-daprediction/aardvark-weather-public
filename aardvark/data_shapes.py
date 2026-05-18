"""
Data is stored in memmaps for access speed. Here we give the shapes required to load these files
"""

CLIMATOLOGY_SHAPE = (4, 366, 24, 240, 121)
CLIMATOLOGY_BASE_SHAPE = (4, 366, 240, 121)


def get_climatology_shape(path):
    """
    Infer climatology_data.mmap shape from file size.

    The original Aardvark setup used 24 upper-air climatology channels. The
    4u_sfc setup can build 30-channel climatology files. Inferring the channel
    count prevents opening a 30-channel memmap with a fixed 24-channel stride.
    """

    import os

    slots, days, x, y = CLIMATOLOGY_BASE_SHAPE
    nbytes = os.path.getsize(path)
    denom = slots * days * x * y * 4
    if nbytes % denom != 0:
        raise ValueError(
            f"Climatology file size is not divisible by expected grid size: {path}"
        )
    channels = nbytes // denom
    return (slots, days, channels, x, y)

ICOADS_Y_SHAPE = (33601, 5, 12000)
ICOADS_X_SHAPE = (33601, 2, 12000)

IGRA_Y_SHAPE = (33604, 24, 1375)
IGRA_X_SHAPE = (1375, 2)

AMSUA_Y_SHAPE = (21916, 180, 360, 13)
AMSUB_Y_SHAPE = (21916, 360, 181, 12)
ASCAT_Y_SHAPE = (21913, 360, 181, 17)
HIRS_Y_SHAPE = (21913, 360, 181, 26)
GRIDSAT_Y_SHAPE = (48211, 2, 514, 200)
IASI_Y_SHAPE = (23373, 360, 181, 52)

# Daily (1D) observation shapes
ICOADS_Y_SHAPE_1D = (4746, 5, 12000)
ICOADS_X_SHAPE_1D = (4746, 2, 12000)

IGRA_Y_SHAPE_1D = (4746, 24, 1375)

AMSUA_Y_SHAPE_1D = (4746, 121, 240, 11)
AMSUB_Y_SHAPE_1D = (4745, 240, 121, 5)
ASCAT_Y_SHAPE_1D = (4746, 240, 121, 15)
HIRS_Y_SHAPE_1D = (4746, 240, 121, 20)
GRIDSAT_Y_SHAPE_1D = (4746, 2, 514, 200)
IASI_Y_SHAPE_1D = (4746, 240, 121, 45)


def get_hadisd_shape(mode):
    """
    Return the shape of the HadISD array depending on variable
    """

    if mode != "train":
        dim_1 = 415
    else:
        var_dict = {"tas": 8719, "tds": 8617, "psl": 8016, "u": 8721, "v": 8721}
        dim_1 = var_dict[var]
    return (106652, dim_1)
