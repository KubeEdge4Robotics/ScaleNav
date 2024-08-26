import argparse
import os
import shutil
import cv2
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

REFERENCE_HEIGHT = 480
REFERENCE_WIDTH = 640


def width_crop(img):
    assert len(img.shape) == 3 and img.shape[2] == 3
    height = img.shape[0]
    width = img.shape[1]
    if height / REFERENCE_HEIGHT > width / REFERENCE_WIDTH:
        target_width = width
        target_height = width * REFERENCE_HEIGHT / REFERENCE_WIDTH
    else:
        target_height = height
        target_width = height * REFERENCE_WIDTH / REFERENCE_HEIGHT

    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(target_width / 2), int(target_height / 2)
    cropped_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return cropped_img

def center_crop(img, ratio):
    assert len(img.shape) == 3 and img.shape[2] == 3
    height = img.shape[0]
    width = img.shape[1]
    target_height = height * ratio
    target_width = width * ratio

    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(target_width / 2), int(target_height / 2)
    cropped_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return cropped_img


def main(args):
    if os.path.exists(args.output_images_dir):
        shutil.rmtree(args.output_images_dir)
    os.makedirs(args.output_images_dir)
    
    image_name_list = os.listdir(args.input_images_dir)
    print(f"image number: {len(image_name_list)}")
    image_name_list = image_name_list[::2]  # downsample from 6Hz to 3Hz
    for image_name in tqdm(image_name_list):
        raw_img = cv2.imread(os.path.join(args.input_images_dir, image_name))
        cv2.imwrite(os.path.join(args.output_images_dir, image_name), raw_img)
    print("finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_images_dir", type=str, help="input images dir")
    parser.add_argument("--output_images_dir", type=str, help="output images dir")
    args = parser.parse_args()
    main(args)
