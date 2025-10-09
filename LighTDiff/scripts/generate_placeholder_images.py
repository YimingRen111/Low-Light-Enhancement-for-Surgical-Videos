#!/usr/bin/env python3
"""
Generate simple placeholder images for smoke tests.

Usage:
  python generate_placeholder_images.py --out1 path/to/bic.png --out2 path/to/gt.png
"""
import argparse
from PIL import Image, ImageDraw, ImageFont
import os

def make_placeholder(path, text, size=(128,128), color=(120,140,200)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new('RGB', size, color)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    w,h = draw.textsize(text, font=font)
    draw.text(((size[0]-w)/2, (size[1]-h)/2), text, fill=(255,255,255), font=font)
    img.save(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out1', required=True)
    parser.add_argument('--out2', required=True)
    args = parser.parse_args()
    make_placeholder(args.out1, 'BIC')
    make_placeholder(args.out2, 'GT')

if __name__ == '__main__':
    main()
