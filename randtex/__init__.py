# todo undo/redo + refactor code using command strategy
# todo save/load sequence of events/commands
# todo save/load set of flats/patches
# todo add similar/disimilar flat (for patch) and patch (for flat)
# todo setup logging


import pathlib

import sys
import io
import random
import configparser

from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie1976
import PySimpleGUI as sg
from omg import *

import numpy as np
import scipy
import scipy.misc
import scipy.cluster
import scipy.cluster
import sklearn.cluster
import numpy
from PIL import Image

import os
import pickle

WINDOW_TITLE = "randtex"

CONFIG_PATH = "randtex.ini"
CACHED_TEXTURE_PACK_SUFFIX = ".pickle"

COMMAND_SEPARATOR = "|"
COMMAND_RANDOMIZE = "Randomize"
COMMAND_CHANGE_SIMILAR = "Change to something similar"
COMMAND_CHANGE_DISIMILAR = "Change to something disimilar"
COMMAND_ADD_SIMILAR = "Add something similar"
COMMAND_ADD_DISIMILAR = "Add something disimilar"
COMMAND_DELETE = "Delete"

HELP_SIMILARITY_FACTOR = """Similarity/Disimilarity factors must be a value between 0.0 and 100.0 (lower numbers mean more similar)
    <= 1.0: Not perceptible by the human eye
    1-2: Perceptible through close observation
    2-10: Perceptible at a glance
    11-49: Colors are more similar than the opposite
    100: Colors are exactly the opposite"""


class Config:
    def __init__(self) -> None:
        self.config = configparser.ConfigParser(defaults={"texture_pack_path": ""})
        self._load()

    def config_path(self):
        return pathlib.Path(__file__).parent.absolute() / CONFIG_PATH

    def _load(self):
        pathlib.Path(self.config_path()).touch()
        self.config.read(self.config_path())

    def save(self):
        with open(self.config_path(), "w") as configfile:
            self.config.write(configfile)

    def set_previous_texture_pack_path(self, texture_pack_path):
        self.config["DEFAULT"]["texture_pack_path"] = texture_pack_path

    def get_previous_texture_pack_path(self):
        return self.config["DEFAULT"]["texture_pack_path"]


def export_to_wad(wad_path, flats, patches):
    wad = WAD(from_file="template/MAP01.wad")
    m = MapEditor(wad.maps["MAP01"])
    z = (0, 0)

    for f, p in zip(flats, patches):
        sector = Sector()
        sector.tx_floor = f
        sector.tx_ceil = "-"
        sidedef = Sidedef()
        sidedef.tx_low = p
        sidedef.tx_up = p
        zx, zy = z
        m.draw_sector(
            [z, (zx + 64, zy), (zx + 64, zy + 64), (zx + 0, zy + 64)],
            sector=sector,
            sidedef=sidedef,
        )
        z = (zx + 64, zy)
    wad.maps["MAP01"] = m.to_lumps()
    wad.to_file(wad_path)


def get_cached_texture_pack_path(path_to_texture_pack):
    prefix = os.path.basename(path_to_texture_pack)
    return f"{prefix}{CACHED_TEXTURE_PACK_SUFFIX}"


def does_cached_texture_pack_exist(path_to_texture_pack):
    return os.path.exists(get_cached_texture_pack_path(path_to_texture_pack))


def show_preview_image_window(im, k):
    d = create_image_data(im, resize_factor=5)
    layout = [
        [sg.Image(data=d, enable_events=True)],
    ]
    window = sg.Window(k, layout)
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            break
        break
    window.close()


def create_image_data(im, resize_factor=0):
    bio = io.BytesIO()
    im = im.copy()
    if resize_factor > 0:
        im = im.resize(
            (im.width * resize_factor, im.height * resize_factor),
            resample=Image.Resampling.NEAREST,
        )
    else:
        im.thumbnail((64, 64))
    im.save(bio, format="PNG")
    return bio.getvalue()


def create_image(im, k):
    d = create_image_data(im)
    return [
        sg.Frame(
            k,
            [
                create_image_buttons_top(k),
                create_image_buttons_bot(k),
                [sg.Image(key=k, data=d, enable_events=True)],
            ],
        )
    ]


def create_image_buttons_top(key):
    return [
        sg.Button(
            "🎲",
            tooltip=COMMAND_RANDOMIZE,
            key=f"{key}{COMMAND_SEPARATOR}{COMMAND_RANDOMIZE}",
        ),
        sg.Button(
            "/∽",
            tooltip=COMMAND_CHANGE_SIMILAR,
            key=f"{key}{COMMAND_SEPARATOR}{COMMAND_CHANGE_SIMILAR}",
        ),
        sg.Button(
            "/≠",
            tooltip=COMMAND_CHANGE_DISIMILAR,
            key=f"{key}{COMMAND_SEPARATOR}{COMMAND_CHANGE_DISIMILAR}",
        ),
    ]


def create_image_buttons_bot(key):
    return [
        sg.Button(
            "+∽",
            tooltip=COMMAND_ADD_SIMILAR,
            key=f"{key}{COMMAND_SEPARATOR}{COMMAND_ADD_SIMILAR}",
        ),
        sg.Button(
            "+≠",
            tooltip=COMMAND_ADD_DISIMILAR,
            key=f"{key}{COMMAND_SEPARATOR}{COMMAND_ADD_DISIMILAR}",
        ),
        sg.Button(
            "X", tooltip=COMMAND_DELETE, key=f"{key}{COMMAND_SEPARATOR}{COMMAND_DELETE}"
        ),
    ]


def cached():
    """
    A function that creates a decorator which will use "cachefile" for caching the results of the decorated function "fn".
    """

    def decorator(fn):  # define a decorator for a function "fn"
        def wrapped(
            *args, **kwargs
        ):  # define a wrapper that will finally call "fn" with all arguments
            # if cache exists -> load it and return its content
            cache_filepath = get_cached_texture_pack_path(args[0])
            if os.path.exists(cache_filepath):
                with open(cache_filepath, "rb") as cachehandle:
                    # print("using cached result from '%s'" % cache_filepath)
                    return pickle.load(cachehandle)

            # execute the function with all arguments passed
            res = fn(*args, **kwargs)

            # write to cache file
            with open(cache_filepath, "wb") as cachehandle:
                # print("saving result to cache '%s'" % cache_filepath)
                pickle.dump(res, cachehandle)

            return res

        return wrapped

    return decorator  # return this "customized" decorator that uses "cachefile"


def show_get_texture_pack_window(texture_pack_path):
    layout = [
        [sg.Text("Choose texture pack to load")],
        [
            sg.Text("Texture Pack", size=(15, 1)),
            sg.InputText(key="-file-", default_text=texture_pack_path),
            sg.FileBrowse(),
        ],
        [sg.Submit(), sg.Cancel()],
    ]

    window = sg.Window(WINDOW_TITLE, layout)
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            sys.exit()
        if values["-file-"] and os.path.exists(values["-file-"]):
            window.close()
            wad = load_wad(values["-file-"])
            return (
                values["-file-"],
                wad,
                list(wad["flats"].keys()),
                list(wad["patches"].keys()),
            )


def find_similar_by_color(
    similar_to_color, current, keys, colors, factor, disimilar=False
):
    if not current:
        return None
    # todo deal with multiple colors at once?

    random.shuffle(keys)
    for k in keys:
        if k not in current:
            d = delta_e_cie1976(similar_to_color, colors[k][0])
            if not disimilar and d < factor or disimilar and d > factor:
                # print(f"d={d} for added flat {f} compared to {similar_to}")
                return k
    return None


def find_similar(similar_to_index, current, keys, colors, factor, disimilar=False):
    # todo deal with multiple colors at once?
    if not current:
        return None
    similar_to = current[similar_to_index]
    similar_to_color = colors[similar_to][0]
    return find_similar_by_color(
        similar_to_color, current, keys, colors, factor, disimilar
    )


def dominant_colors(image):  # PIL image input
    image = image.resize((150, 150))  # optional, to reduce time
    ar = numpy.asarray(image)
    shape = ar.shape
    ar = ar.reshape(numpy.product(shape[:2]), shape[2]).astype(float)

    kmeans = sklearn.cluster.MiniBatchKMeans(
        n_clusters=10, n_init=3, init="k-means++", max_iter=20, random_state=1000
    ).fit(ar)
    codes = kmeans.cluster_centers_

    vecs, _dist = scipy.cluster.vq.vq(ar, codes)  # assign codes
    counts, _bins = numpy.histogram(vecs, len(codes))  # count occurrences

    colors = []
    for index in numpy.argsort(counts)[::-1]:
        colors.append(tuple([int(code) for code in codes[index]]))
    return colors  # returns colors in order of dominance


@cached()
def load_wad(wad_path):
    wad = WAD(from_file=wad_path)
    from_flat_to_image = {}
    for k in wad.flats:
        from_flat_to_image[k] = wad.flats[k].to_Image(mode="RGBA")
    from_flat_to_dominant_colors = {}
    for k in wad.flats:
        from_flat_to_dominant_colors[k] = [
            convert_color(sRGBColor(r, g, b, is_upscaled=True), LabColor)
            for r, g, b, _ in dominant_colors(from_flat_to_image[k])
        ]
    from_patch_to_image = {}
    for k in wad.patches:
        from_patch_to_image[k] = wad.patches[k].to_Image(mode="RGBA")
    from_patch_to_dominant_colors = {}
    for k in wad.patches:
        from_patch_to_dominant_colors[k] = [
            convert_color(sRGBColor(r, g, b, is_upscaled=True), LabColor)
            for r, g, b, _ in dominant_colors(from_patch_to_image[k])
        ]
    return {
        "flats": from_flat_to_dominant_colors,
        "patches": from_patch_to_dominant_colors,
        "flat_images": from_flat_to_image,
        "patch_images": from_patch_to_image,
    }


EVENT_ADD_RANDOM_FLAT = "Add Random Flat"
EVENT_ADD_RANDOM_PATCH = "Add Random Patch​"
EVENT_RESET = "Reset All"
EVENT_EXPORT_TEMPLATE_WAD = "Export template wad"


def main():
    skip_window_close = False
    window = None
    config = Config()
    flats = []
    patches = []
    similarity_factor = 20
    disimilarity_factor = 40
    texture_pack_path, wad, all_flats, all_patches = show_get_texture_pack_window(
        config.get_previous_texture_pack_path()
    )
    config.set_previous_texture_pack_path(texture_pack_path)
    config.save()
    while True:
        flat_images = []
        for f in flats:
            im = wad["flat_images"][f]
            flat_images += create_image(im, f)
        patch_images = []
        for p in patches:
            im = wad["patch_images"][p]
            patch_images += create_image(im, p)
        layout = [
            [
                sg.Text(
                    f"Texture Pack: {os.path.basename(texture_pack_path)} | # Flats: {len(all_flats)} | # Patches: {len(all_patches)}"
                )
            ],
            [sg.Text(f"- Click on a flat/patch to see a larger version of it")],
            [sg.Text(f"- Right-click on a flat/patch to access features")],
            [sg.Text(f"{HELP_SIMILARITY_FACTOR}")],
            [
                sg.Text("Similarity Factor"),
                sg.InputText(
                    str(similarity_factor), key="-similarity_factor-", size=(5, 1)
                ),
                sg.Text("Disimilarity Factor"),
                sg.InputText(
                    str(disimilarity_factor),
                    key="-disimilarity_factor-",
                    size=(5, 1),
                ),
                sg.Button(EVENT_RESET),
                sg.Button(EVENT_EXPORT_TEMPLATE_WAD),
            ],
            [
                sg.Text("Flats"),
                sg.Button(EVENT_ADD_RANDOM_FLAT),
            ],
            flat_images if flat_images else [sg.Text("\n\n\n\n\n\n\n\n")],
            [
                sg.Text("Patches"),
                sg.Button(EVENT_ADD_RANDOM_PATCH),
            ],
            patch_images if patch_images else [sg.Text("\n\n\n\n\n\n\n\n")],
        ]

        old_window_location = (
            window.current_location() if window is not None else (None, None)
        )
        if window is not None and not skip_window_close:
            skip_window_close = False
            window.close()
        window = sg.Window(WINDOW_TITLE, layout, location=old_window_location)
        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            break

        tex = None
        if event and COMMAND_SEPARATOR in event:
            tex, event = event.split(COMMAND_SEPARATOR, 1)
        try:
            similarity_factor = float(values["-similarity_factor-"])
            if similarity_factor < 0.0 or similarity_factor >= 100.0:
                raise ValueError
        except ValueError:
            sg.popup_error(HELP_SIMILARITY_FACTOR)
            continue
        try:
            disimilarity_factor = float(values["-disimilarity_factor-"])
            if disimilarity_factor < 0.0 or disimilarity_factor >= 100.0:
                raise ValueError
        except ValueError:
            sg.popup_error(HELP_SIMILARITY_FACTOR)
            continue
        if event == EVENT_ADD_RANDOM_FLAT:
            f = all_flats[random.randint(0, len(all_flats) - 1)]
            # print(f"Adding flat {f}")
            flats += [f]
        if event == EVENT_ADD_RANDOM_PATCH:
            p = all_patches[random.randint(0, len(all_patches) - 1)]
            patches += [p]
            # print(f"Adding patch {p}")
        if event == EVENT_RESET:
            flats = []
            patches = []
        if event == EVENT_EXPORT_TEMPLATE_WAD:
            # todo add a filebrowser for this
            export_to_wad("test.wad", flats, patches)

        if event == COMMAND_DELETE:
            try:
                flats.remove(tex)
            except ValueError:
                pass
            try:
                patches.remove(tex)
            except ValueError:
                pass
        if event == COMMAND_RANDOMIZE:
            try:
                i = flats.index(tex)
                flats[i] = all_flats[random.randint(0, len(all_flats) - 1)]
            except ValueError:
                pass
            try:
                i = patches.index(tex)
                patches[i] = all_patches[random.randint(0, len(all_patches) - 1)]
            except ValueError:
                pass
        if event == COMMAND_ADD_SIMILAR or event == COMMAND_CHANGE_SIMILAR:
            try:
                i = flats.index(tex)
                f = find_similar(
                    i,
                    flats,
                    all_flats,
                    wad["flats"],
                    similarity_factor,
                    disimilar=False,
                )
                if f:
                    if event == COMMAND_ADD_SIMILAR:
                        flats += [f]
                    else:
                        flats[i] = f
            except ValueError:
                pass
            try:
                i = patches.index(tex)
                p = find_similar(
                    i,
                    patches,
                    all_patches,
                    wad["patches"],
                    similarity_factor,
                    disimilar=False,
                )
                if p:
                    if event == COMMAND_ADD_SIMILAR:
                        patches += [p]
                    else:
                        patches[i] = p
            except ValueError:
                pass

        if event == COMMAND_ADD_DISIMILAR or event == COMMAND_CHANGE_DISIMILAR:
            try:
                i = flats.index(tex)
                f = find_similar(
                    i,
                    flats,
                    all_flats,
                    wad["flats"],
                    disimilarity_factor,
                    disimilar=True,
                )
                if f:
                    if event == COMMAND_ADD_DISIMILAR:
                        flats += [f]
                    else:
                        flats[i] = f
            except ValueError:
                pass
            try:
                i = patches.index(tex)
                p = find_similar(
                    i,
                    patches,
                    all_patches,
                    wad["patches"],
                    disimilarity_factor,
                    disimilar=True,
                )
                if p:
                    if event == COMMAND_ADD_DISIMILAR:
                        patches += [p]
                    else:
                        patches[i] = p
            except ValueError:
                pass
        if flats and event in flats:
            # Clicking on an flat
            im = wad["flat_images"][event]
            show_preview_image_window(im, event)
            skip_window_close = True
        if patches and event in patches:
            # Clicking on an patch
            im = wad["patch_images"][event]
            show_preview_image_window(im, event)
            skip_window_close = True


if __name__ == "__main__":
    main()
