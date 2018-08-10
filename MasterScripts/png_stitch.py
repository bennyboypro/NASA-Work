import sys
from PIL import Image
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--inputs', nargs='+',
                    help="Input png files")
parser.add_argument('--output', type=str,
                    help="Output png file")

params = parser.parse_args()

images = map(Image.open, params.inputs)
widths, heights = zip(*(i.size for i in images))

total_width = sum(widths)
max_height = max(heights)

new_im = Image.new('RGB', (total_width, max_height))

x_offset = 0
for im in images:
    new_im.paste(im, (x_offset, 0))
    x_offset += im.size[0]

new_im.save(params.output)