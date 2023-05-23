import sys
import io
import random

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


def cached(cachefile):
    """
    A function that creates a decorator which will use "cachefile" for caching the results of the decorated function "fn".
    """

    def decorator(fn):  # define a decorator for a function "fn"
        def wrapped(
            *args, **kwargs
        ):  # define a wrapper that will finally call "fn" with all arguments
            # if cache exists -> load it and return its content
            prefix = os.path.basename(args[0])
            cachefile_with_prefix = f"{prefix}{cachefile}"
            if os.path.exists(cachefile_with_prefix):
                with open(cachefile_with_prefix, "rb") as cachehandle:
                    print("using cached result from '%s'" % cachefile_with_prefix)
                    return pickle.load(cachehandle)

            # execute the function with all arguments passed
            res = fn(*args, **kwargs)

            # write to cache file
            with open(cachefile_with_prefix, "wb") as cachehandle:
                print("saving result to cache '%s'" % cachefile_with_prefix)
                pickle.dump(res, cachehandle)

            return res

        return wrapped

    return decorator  # return this "customized" decorator that uses "cachefile"


def find_similar(similar_to_index, current, keys, colors, factor, disimilar=False):
    if not current:
        return None
    similar_to = current[similar_to_index]
    # todo deal with multiple colors at once?
    similar_to_color = colors[similar_to][0]
    random.shuffle(keys)
    for f in keys:
        if f not in current:
            d = delta_e_cie1976(similar_to_color, colors[f][0])
            if not disimilar and d < factor or disimilar and d > factor:
                print(f"d={d} for added flat {f} compared to {similar_to}")
                return f
    return None


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


# todo this should cache based on the passed in file
@cached(".pickle")
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


EVENT_ADD_RANDOM_FLAT = "Add"
EVENT_ADD_RANDOM_PATCH = "Add​"
EVENT_RESET = "Reset All"


def main():
    wad_loaded = False
    wad = None
    all_flats = None
    all_patches = None
    flats = []
    patches = []
    similarity_factor = 20
    disimilarity_factor = 40
    if not wad_loaded:
        print("loading...")
        flats = []
        patches = []
        wad_loaded = True
        wad = load_wad(r"C:\Users\Bill Tyros\Documents\GitHub\randtex\tex\OTEX_1.1.WAD")
        all_flats = list(wad["flats"].keys())
        all_patches = list(wad["patches"].keys())
    while True:
        flat_images = []
        for f in flats:
            im = wad["flat_images"][f]
            bio = io.BytesIO()
            im = im.copy()
            im.thumbnail((128, 128))
            im.save(bio, format="PNG")
            flat_images += [
                sg.Frame(
                    f,
                    [
                        create_image_buttons_top(f),
                        create_image_buttons_bot(f),
                        [sg.Image(key=f, data=bio.getvalue())],
                    ],
                )
            ]
        patch_images = []
        for p in patches:
            im = wad["patch_images"][p]
            bio = io.BytesIO()
            im = im.copy()
            im.thumbnail((128, 128))
            im.save(bio, format="PNG")
            patch_images += [
                sg.Frame(
                    p,
                    [
                        create_image_buttons_top(p),
                        create_image_buttons_bot(p),
                        [sg.Image(key=p, data=bio.getvalue())],
                    ],
                )
            ]
        layout = [
            [
                sg.Text(
                    f"- Right-click on a flat/patch to access features\n- {HELP_SIMILARITY_FACTOR}"
                )
            ],
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
            ],
            [
                sg.Text("Flats"),
                sg.Button(EVENT_ADD_RANDOM_FLAT),
            ],
            flat_images,
            [
                sg.Text("Patches"),
                sg.Button(EVENT_ADD_RANDOM_PATCH),
            ],
            [patch_images],
        ]

        # Create the window
        window = sg.Window("randtex", layout)

        event, values = window.read()

        tex = None
        if event and COMMAND_SEPARATOR in event:
            tex, event = event.split(COMMAND_SEPARATOR, 1)
        print(tex)
        print(event)
        # See if user wants to quit or window was closed
        if event == sg.WINDOW_CLOSED:
            break

        try:
            similarity_factor = float(values["-similarity_factor-"])
            if similarity_factor < 0.0 or similarity_factor >= 100.0:
                raise ValueError
        except ValueError:
            sg.popup_error(HELP_SIMILARITY_FACTOR)
            window.close()
            continue
        try:
            disimilarity_factor = float(values["-disimilarity_factor-"])
            if disimilarity_factor < 0.0 or disimilarity_factor >= 100.0:
                raise ValueError
        except ValueError:
            sg.popup_error(HELP_SIMILARITY_FACTOR)
            window.close()
            continue
        if event == EVENT_ADD_RANDOM_FLAT:
            f = all_flats[random.randint(0, len(all_flats))]
            print(f"Adding flat {f}")
            flats += [f]
        if event == EVENT_ADD_RANDOM_PATCH:
            p = all_patches[random.randint(0, len(all_patches))]
            patches += [p]
            print(f"Adding patch {p}")
        if event == EVENT_RESET:
            flats = []
            patches = []

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
                flats[i] = all_flats[random.randint(0, len(all_flats))]
            except ValueError:
                pass
            try:
                i = patches.index(tex)
                patches[i] = all_patches[random.randint(0, len(all_patches))]
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
        # Finish up by removing from the screen
        window.close()


if __name__ == "__main__":
    main()
