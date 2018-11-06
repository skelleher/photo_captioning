#/usr/bin/env python

# For every folder in destination:
#   for every file in folder:
#       query the caption server
#       display the photo (optional) and caption

import os
import sys
import argparse
import math
from stat import *
import requests
import json
from PIL import Image, ImageFont, ImageDraw

_args = None

def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="folder or file to query.  Will be captioned recursively")
    parser.add_argument("--host", help="hostname of query server", nargs="?", default="localhost")
    parser.add_argument("--port", help="port of query server", nargs="?", type=int, default=1975)
    parser.add_argument("--show", help="show query results in a popup window", action="store_true")

    global _args
    _args = parser.parse_args()
    path = _args.input

    # Remove trailing slash from path
    path = path.rstrip(os.path.sep)

    # Check if input path exists.
    if not os.path.exists(path):
        print("Error: %s not found" % path)
        return -1

    filenames, captions = _caption(path, _args)

    # Flatten the lists of lists
    filenames_flat = [item for sublist in filenames for item in sublist]
    captions_flat  = [item for sublist in captions for item in sublist]

    for name, caption in zip(filenames_flat, captions_flat):
        print("%s = %s" % (name, caption))
   
    if _args.show:
        _show_images(filenames_flat, captions_flat)

    return


def _caption(path, args):
    # Check if input path exists.
    if not os.path.exists(path):
        print("Error: %s not found" % path)
        return -1

    filenames = []
    captions = []
    if os.path.isfile(path):
        name, caption = _caption_file(path, args)
        filenames.append(name)
        captions.append(caption)
    elif os.path.isdir(path):
        names, caps = _caption_folder(path, args)

        filenames.append(names)
        captions.append(caps)

    return filenames, captions


# Recursively calls itself for all subfolders
def _caption_folder(input_path, args):
    filenames = []
    captions = []

    for name in os.listdir(input_path):
        path = input_path + os.path.sep + name

        if name[0] == '.':
            continue

        if os.path.isfile(path):
            name, caption = _caption_file(path, args)
            filenames.append(name)
            captions.append(caption)

        if os.path.isdir(path):
            names, captions = _caption_folder(path, args)
            filenames.append(names)
            captions.append(captions)

    return filenames, captions


def _caption_file(input_path, args):
    url = "http://%s:%d/query" % (args.host, args.port)
    headers = { "content-type" : "application/octet-stream" }

    try:
        payload = open(input_path, "rb").read()
    except Exception as ex:
        print("Error loading file %s" % input_path)
        print(type(ex))
        print(ex.args)
        print(ex)
        pass

    reply = requests.post(url, data=payload, headers=headers)
    caption = reply.text.strip()

    print("%s = %s" % (input_path, caption))

    return input_path, caption


def _factor_pairs( integer ):
    return [((int)(x), (int)(integer/x)) for x in range(1, int(math.sqrt(integer))+1) if integer % x == 0]


# return best grid size for a list of <integer> images 
# e.g. a list of N images is best shown as a grid of X x Y
def _grid_size( integer ):
    pairs = _factor_pairs( integer )
    pair = pairs[ -1 ]
    print("pairs = ", pairs)
    
    # Optimize for wider images, and more rows
    return max(pair), min(pair)


def _show_images(filenames, captions = None):
    num_files   = len(filenames)
    height      = 320 
    width       = 320
    pad_w       = 10
    pad_h       = 10

    grid_width_items, grid_height_items = _grid_size( num_files )
    #print("grid (%d) = %d x %d" % (num_files, grid_width_items, grid_height_items))

    grid_width  = int(pad_w + (grid_width_items * (width + pad_w)) + pad_w)
    grid_height = int(pad_h + (grid_height_items * (height + pad_h)) + pad_h)
    #print("grid = %d x %d" % (grid_width, grid_height))

    grid = Image.new("RGBA", (grid_width, grid_height), color=(255,255,255,0))

    for y in range(grid_height_items):
        for x in range(grid_width_items):
            i = y * grid_width_items + x
            #print("%d, %d = %d" % (x, y, i))
            path = filenames[ i ]
            caption = captions[ i ]

            with Image.open( path ) as image:
                image = image.resize((height, width), resample = Image.BILINEAR)
                draw = ImageDraw.Draw(image)
                draw.rectangle(((0, 0), (width, 20)), fill="white")
                draw.text((2, 4), caption, (0,0,0))

                left = x * width + pad_w
                right = y * height + pad_h
                grid.paste(image, box = (left, right))

    grid.show()


 
if __name__ == "__main__":
    _main()

