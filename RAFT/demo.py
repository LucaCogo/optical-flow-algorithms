import sys
sys.path.append('core')

import argparse
import os
import time
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder


DEVICE = 'cpu'
FRAMES_PATH = ''
OUT_FOLDER = ''

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, count = ""):

    img = img.permute(1,2,0).cpu().numpy()
    flo = flo.permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)
   
    img_flo = cv2.cvtColor(img_flo, cv2.COLOR_RGB2BGR)


    path = FRAMES_PATH + "/" + OUT_FOLDER + "-estimates" 
    if not os.path.exists(path):
        os.mkdir(path)
       
    imname =  OUT_FOLDER + "_frame_" + str(count) + ".png"
    filename = os.path.join(path, imname)
    cv2.imwrite(filename, img_flo)
    print(f"Saved to {filename}")



def demo(args):

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=torch.device(DEVICE)))

    model = model.module
    model.to(DEVICE)
    model.eval()

    t = time.time()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        count = 0
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
       
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

            if OUT_FOLDER is not None:
                viz(image1[0], flow_up[0], count)

            count +=1
    print(f"Elapsed time (iterative): {time.time() - t}")

def demo_batch(args):
    
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=torch.device(DEVICE)))

    model = model.module
    model.to(DEVICE)
    model.eval()

    t = time.time()
    with torch.no_grad():
        image_filenames = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        image_filenames = sorted(image_filenames)

        images = []
        for imfile in image_filenames:
                images.append(load_image(imfile))

        images = torch.cat(images)



        padder = InputPadder(images.shape)
        images = padder.pad(images)[0]


        images_batch1 = images[:-1]
        images_batch2 = images[1:]

        
        _, flow_out = model(images_batch1, images_batch2, iters=20, test_mode=True)

        # images = padder.unpad(images)


        if OUT_FOLDER is not None:
            for count, image in enumerate(images[:-1]):
                viz(image, flow_out[count], count) 

    print(f"Elapsed time (batch): {time.time() - t}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--batch_mode', action='store_true')
    parser.add_argument('--output_folder', help="name of folder where to save results")
    args = parser.parse_args()

    FRAMES_PATH = args.path
    OUT_FOLDER = args.output_folder

    if args.batch_mode:
        print("Batch mode")
        demo_batch(args)
    else:
        print("Iterative mode")
        demo(args)
