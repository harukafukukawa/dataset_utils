import argparse
import glob
from PIL import Image

help_description = "Converts 3 channel RGB images to 1 channel greyscale images"
# Create the parser
parser = argparse.ArgumentParser(description=help_description)

# Add arguments to the parser
parser.add_argument('--path', type=str, help='Path to directory where images are (images will be found recursively)')
parser.add_argument('--in_place', type=bool, default=True, help='If true, the images will be overwritten in place, default=`True`')

# Parse the command line arguments
args = parser.parse_args()
# Access the arguments
if args.in_place == False:
    raise NotImplementedError("Out of place has not yet been implemented")
    

paths = glob.glob(args.path + "/**", recursive=True)

for path in paths:
    if path.endswith('.png'):
        img = Image.open(path).convert('L')
        img.save(path)