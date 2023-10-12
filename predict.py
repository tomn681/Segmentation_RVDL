import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from utils.dataset import DefaultDataset
from utils.dice_score import multiclass_dice_coeff, dice_coeff

import models.unet as unet

	     
model_dict = {'U-Net': unet.UNet,
	     }
	     
def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[1, :, :])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()

def predict_img(net,
                full_img,
                mask_true, #############
                device,
                dataset,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(dataset.preprocess(full_img, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        full_mask = (full_mask > out_threshold)
    else:
        full_mask = F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1)
        
    mask_true = torch.from_numpy(np.asarray(mask_true, dtype=np.uint8)) #########
    mask_true = mask_true.unsqueeze(0) #########
    threshold = (torch.max(mask_true) - torch.min(mask_true))//2 ############
    mask_true = (mask_true > threshold) * 1 ##############
    tf_mask = transforms.Resize((full_img.size[1], full_img.size[0]))
    mask_true = tf_mask(mask_true)
    mask_true = torch.concat((mask_true, mask_true), dim=0)
    #print(mask_true.shape)
    #
    #print(full_mask.shape)

    dice_score = multiclass_dice_coeff(full_mask, mask_true, reduce_batch_first=False) #########
    
    return full_mask.numpy(), dice_score ############# CORREGIR #############


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='U-Net', metavar='FILE',
                        help='Specify the model used')
    parser.add_argument('--pth', '-p', default='MODEL', metavar='FILE',
                        help='Specify the pth filename')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    
    # Image Info
    parser.add_argument('--file-path', '-fp', metavar='PTH', type=str, default='./data/Montgomery-Shenzhen/', help='train file path', dest='path')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)
    
    model = model_dict[args.model]
    
    net = model(n_channels=1, n_classes=2, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.pth + '.pth', map_location=device))

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'\nPredicting image {filename} ...')
        img = Image.open(filename)
        mask_true = Image.open(filename[:-4]+'_mask.png') #########
        
        dataset = DefaultDataset(args.path)

        mask, dice_score = predict_img(net=net,
                           full_img=img,
                           mask_true=mask_true, #######
                           dataset = dataset,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
       

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask)
            result.save(out_filename + f' (DSC:{dice_score}).png')  ######### CORREGIR #########
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
