import os
import sys
import numpy as np
import torch
import time
import argparse
import torch.utils.data
import torch.nn.functional as F
from PIL import Image
import kornia

from LMN_model import *
from DNN_model import DNN_Network
from CAN_model import *
from JPG_multi_read_data import MemoryFriendlyLoader


parser = argparse.ArgumentParser("Ghillie")
parser.add_argument('--data_path', type=str, default='./data/LIME')
parser.add_argument('--save_path', type=str, default='./results/LIME')
parser.add_argument('--LMN_model', type=str, default='./weights/LOL-V2-Syn/PSNR/LMN_weights.pt')
parser.add_argument('--DNN_model', type=str, default='./weights/LOL-V2-Syn/PSNR/DNN_weights.pt')
parser.add_argument('--CAN_model', type=str, default='./weights/LOL-V2-Syn/PSNR/CAN_weights.pt')
# parser.add_argument('--data_path', type=str, default='', help='location of the data corpus')
# parser.add_argument('--save_path', type=str, default='', help='location of the data corpus')
# parser.add_argument('--LMN_model', type=str, default='./weights/MIT/PSNR/LMN_weights.pt', help='location of the data corpus')
# parser.add_argument('--DNN_model', type=str, default='./weights/MIT/PSNR/DNN_weights.pt', help='location of the data corpus')
# parser.add_argument('--CAN_model', type=str, default='', help='location of the data corpus')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--M', type=int, default=32)

args = parser.parse_args()
os.makedirs(args.save_path, exist_ok=True)


# -------------------------------------------------------------
# Utils
# -------------------------------------------------------------
def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')


# LMN has 3 × PixelUnshuffle → 必须 8 的倍数
def padding(image, divide_size=8):
    n, c, h, w = image.shape
    pad_h = (divide_size - h % divide_size) % divide_size
    pad_w = (divide_size - w % divide_size) % divide_size
    image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    return image, h, w


def unpadding(image, h, w):
    return image[:, :, :h, :w]


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    if not torch.cuda.is_available():
        print("No GPU available.")
        sys.exit(1)

    # Load Models
    LMN_model = Network(args.batch_size, args.M).cuda()
    LMN_model.load_state_dict(torch.load(args.LMN_model))

    DNN_model = DNN_Network().cuda()
    DNN_model.load_state_dict(torch.load(args.DNN_model))

    CAN_model = CAN_Network().cuda()
    CAN_model.load_state_dict(torch.load(args.CAN_model))

    # Load Data
    TestDataset = MemoryFriendlyLoader(img_dir=args.data_path, task='test')
    test_queue = torch.utils.data.DataLoader(
        TestDataset, batch_size=1, pin_memory=True, num_workers=0)

    with torch.no_grad():
        Runtime = []

        for _, (input, image_name) in enumerate(test_queue):

            input = input.cuda()

            # ---- padding for PixelUnshuffle ×3 ----
            input, ori_h, ori_w = padding(input, divide_size=8)

            image_name = image_name[0].split('/')[-1].split('.')[0]

            # ---- Inference ----
            start = time.perf_counter()

            x, y_en, s = LMN_model(input)
            _, _, _, _, y_en = DNN_model(y_en)
            out_cbcr = CAN_model(input, y_en)

            cb, cr = torch.split(out_cbcr, 1, dim=1)
            out_img = kornia.color.ycbcr_to_rgb(torch.cat([y_en, cb, cr], dim=1))

            # ---- unpadding ----
            out_img = unpadding(out_img, ori_h, ori_w)

            end = time.perf_counter()
            Runtime.append(end - start)

            # ---- Save ----
            save_path = os.path.join(args.save_path, image_name + ".png")
            print(f"processing {image_name}.png")
            save_images(out_img, save_path)

        print("\n==== Runtime Statistics ====")
        mean_rt = sum(Runtime) / len(Runtime)
        var_rt = sum((x - mean_rt) ** 2 for x in Runtime) / len(Runtime)
        print(f"Mean runtime: {mean_rt}")
        print(f"Variance:     {var_rt}")


if __name__ == '__main__':
    main()
