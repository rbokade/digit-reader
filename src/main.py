import os
import glob
import argparse
import warnings

import cv2
import torch
import pytesseract
import numpy as np
import pandas as pd

from PIL import Image
from torchvision import transforms

from model import Model
from utils.video_processor import VideoProcessor
from bounding_box import BoundingBoxWidget

warnings.filterwarnings("ignore")


dirname = os.path.dirname(__file__)
__model_path__ = os.path.join(dirname, "model.pth")


def get_reference_image(images_dir):
    images_filenames = sorted(glob.glob(images_dir + "*.jpg"))
    return images_filenames[0]


def crop_and_save_images(images_dir, bounding_box):
    images_filenames = sorted(glob.glob(images_dir + "*.jpg"))
    for img_path in images_filenames:
        image = Image.open(img_path).convert("RGB")
        try:
            cropped_image = image.crop((
                bounding_box[0][0],
                bounding_box[0][1],
                bounding_box[1][0],
                bounding_box[1][1]
            ))
            cropped_image.save(img_path)
        except:
            import pdb; pdb.set_trace()


def cv2_process_image(img):
    """
    alpha scale: 1-3
    brightness scale: 0-100
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=20)
    img = cv2.medianBlur(img, 3)
    img = cv2.threshold(img, 135, 255, cv2.THRESH_BINARY)[1]
    return img


def pytorch_process_image(image):
    transform = transforms.Compose([
            transforms.Resize([54, 54], interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    pytorch_image = transform(image)
    return pytorch_image


def preprocess_images(images_dir):
    images_filenames = sorted(glob.glob(images_dir + "*.jpg"))
    normalized_images = []
    for img_file in images_filenames:
        img = cv2_process_image(cv2.imread(img_file))
        img = Image.fromarray(img)
        if len(img.size) < 3:
            img = img.convert("RGB")
        normalized_img = pytorch_process_image(img)
        normalized_images.append(normalized_img)
    return torch.stack(normalized_images)


# def process_with_tesseract(images_dir, output_dir, output_filename):
#     images_filenames = sorted(glob.glob(images_dir + "*.jpg"))
#     outs = {"sec": [], "digit": []}
#     for img_file in images_filenames:
#         img = cv2.imread(img_file)
#         #convert to grayscale image
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img = cv2.convertScaleAbs(img, alpha=1.5, beta=20)
#         img = cv2.medianBlur(img, 3)
#         img = cv2.threshold(img, 135, 255, cv2.THRESH_BINARY)[1]
#         # img = cv2.medianBlur(img, 5)
#         # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#         cv2.imwrite(img_file, img)
#         text = pytesseract.image_to_string(Image.open(img_file), config='outputbase digits')
#         frame_name = img_file.split("/")[-1].split(".")[0].split("_")[0]
#         outs["sec"].append(float(frame_name))
#         outs["digit"].append(text)
#     df = pd.DataFrame.from_dict(outs).to_csv(output_dir + output_filename + ".csv", sep=",")


def process_output(images_dir, model_out):
    images_filenames = sorted(glob.glob(images_dir + "*.jpg"))
    model_out_dict = {"sec": [], "digits": []}
    for idx, img_file in enumerate(images_filenames):
        model_out_dict["sec"].append(
            float(img_file.split("/")[-1].split(".")[0].split("_")[0])
        )
        pred_len = model_out[0, idx].item()
        prediction = ""
        for l in range(pred_len):
            digit = model_out[l + 1, idx].item()
            if digit != 10:
                prediction += str(digit)
        model_out_dict["digits"].append(float(prediction))
    return pd.DataFrame.from_dict(model_out_dict).sort_values(by="sec")


def save_as_csv(processed_output, output_dir, output_filename):
    if not output_dir[-1] == "/":
        output_dir += "/"
    processed_output.to_csv(output_dir + output_filename + ".csv", sep=",")


def main(args):
    # Extract images
    video_processor = VideoProcessor(args.video_dir, args.output_dir, downscale=False)
    video_processor.extract_all_frames_from_video(capture_interval=10, save_frames=True)
    _tmp_output_dir = video_processor._output_frames_dir
    # Bounding box and image preprocessing
    reference_image = get_reference_image(_tmp_output_dir)
    boundingbox_widget = BoundingBoxWidget(reference_image)
    bounding_box = boundingbox_widget.get_bounding_box()
    crop_and_save_images(_tmp_output_dir, bounding_box)
    normalized_images = preprocess_images(_tmp_output_dir)
    # Model inference and post processing
    model = Model()
    model.load_model(__model_path__)
    model_out = model(normalized_images)
    processed_output = process_output(_tmp_output_dir, model_out)
    # Save as csv
    save_as_csv(processed_output, args.output_dir, args.output_filename)
    # process_with_tesseract(_tmp_output_dir, args.output_dir, args.output_filename)


if __name__ == "__main__":

    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video-dir", dest="video_dir", type=str,
        required=True, help="Path to the video file"
    )
    parser.add_argument(
        "--output-dir", dest="output_dir", type=str,
        required=True, help="Path to save logs"
    )
    parser.add_argument(
        "--output-filename", dest="output_filename", type=str,
        required=True, help="Output file name"
    )

    args = parser.parse_args()

    main(args)
