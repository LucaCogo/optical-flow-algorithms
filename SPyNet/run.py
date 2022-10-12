#!/usr/bin/env python

# import getopt
import argparse
import glob
import math
import numpy as np
import cv2
import PIL
import PIL.Image
import os
import sys
import torch
import flow_viz
import time

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

DEVICE = 'cpu'

MODEL = ''

##########################################################

# arguments_strModel = 'sintel-final' # 'sintel-final', or 'sintel-clean', or 'chairs-final', or 'chairs-clean', or 'kitti-final'


# arguments_strOne = './images/one.png'
# arguments_strTwo = './images/two.png'
# arguments_strOut = './out.flo'
# FRAMES_PATH = ''

# for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
#     if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use, see below
#     if strOption == '--one' and strArgument != '': arguments_strOne = strArgument # path to the first frame
#     if strOption == '--two' and strArgument != '': arguments_strTwo = strArgument # path to the second frame
#     if strOption == '--path' and strArgument != '': FRAMES_PATH = strArgument
#     if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
# end

##########################################################

backwarp_tenGrid = {}

def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1)
    # end

    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

    return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=False)
# end

##########################################################

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        class Preprocess(torch.nn.Module):
            def __init__(self):
                super().__init__()
            # end

            def forward(self, tenInput):
                tenInput = tenInput.flip([1])
                tenInput = tenInput - torch.tensor(data=[0.485, 0.456, 0.406], dtype=tenInput.dtype, device=tenInput.device).view(1, 3, 1, 1)
                tenInput = tenInput * torch.tensor(data=[1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225], dtype=tenInput.dtype, device=tenInput.device).view(1, 3, 1, 1)

                return tenInput
            # end
        # end

        class Basic(torch.nn.Module):
            def __init__(self, intLevel):
                super().__init__()

                self.netBasic = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
                )
            # end

            def forward(self, tenInput):
                return self.netBasic(tenInput)
            # end
        # end

        self.netPreprocess = Preprocess()

        self.netBasic = torch.nn.ModuleList([ Basic(intLevel) for intLevel in range(6) ])

        self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-spynet/network-' + MODEL + '.pytorch', file_name='spynet-' + MODEL).items() })
    # end

    def forward(self, tenOne, tenTwo):
        tenFlow = []

        tenOne = [ self.netPreprocess(tenOne) ]
        tenTwo = [ self.netPreprocess(tenTwo) ]

        for intLevel in range(5):
            if tenOne[0].shape[2] > 32 or tenOne[0].shape[3] > 32:
                tenOne.insert(0, torch.nn.functional.avg_pool2d(input=tenOne[0], kernel_size=2, stride=2, count_include_pad=False))
                tenTwo.insert(0, torch.nn.functional.avg_pool2d(input=tenTwo[0], kernel_size=2, stride=2, count_include_pad=False))
            # end
        # end

        tenFlow = tenOne[0].new_zeros([ tenOne[0].shape[0], 2, int(math.floor(tenOne[0].shape[2] / 2.0)), int(math.floor(tenOne[0].shape[3] / 2.0)) ])

        for intLevel in range(len(tenOne)):
            tenUpsampled = torch.nn.functional.interpolate(input=tenFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            if tenUpsampled.shape[2] != tenOne[intLevel].shape[2]: tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[ 0, 0, 0, 1 ], mode='replicate')
            if tenUpsampled.shape[3] != tenOne[intLevel].shape[3]: tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[ 0, 1, 0, 0 ], mode='replicate')

            tenFlow = self.netBasic[intLevel](torch.cat([ tenOne[intLevel], backwarp(tenInput=tenTwo[intLevel], tenFlow=tenUpsampled), tenUpsampled ], 1)) + tenUpsampled
        # end

        return tenFlow
    # end
# end

netNetwork = None

##########################################################

def estimate(tenOne, tenTwo):

    global netNetwork

    if netNetwork is None:
        netNetwork = Network().eval()
    # end

    assert(tenOne.shape[2] == tenTwo.shape[2])
    assert(tenOne.shape[3] == tenTwo.shape[3])

    intWidth = tenOne.shape[3]
    intHeight = tenOne.shape[2]

    # assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    # assert(intHeight == 416) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    # tenPreprocessedOne = tenOne.view(1, 3, intHeight, intWidth)
    # tenPreprocessedTwo = tenTwo.view(1, 3, intHeight, intWidth)

    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

    tenPreprocessedOne = torch.nn.functional.interpolate(input=tenOne, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
    tenPreprocessedTwo = torch.nn.functional.interpolate(input=tenTwo, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

    tenFlow = torch.nn.functional.interpolate(input=netNetwork(tenPreprocessedOne, tenPreprocessedTwo), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

    tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    return tenFlow[:, :, :, :].cpu()
# end

##########################################################

def load_image(imfile):
    img = np.ascontiguousarray(np.array(PIL.Image.open(imfile))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))
    img = torch.FloatTensor(img)

    return img.to(DEVICE)

def viz(img, flo, count = ""):
    img = img.permute(1,2,0).cpu().numpy()
    img *= 255
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    flo = flo.permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)
   
    img_flo = cv2.cvtColor(img_flo, cv2.COLOR_RGB2BGR)


    path = FRAMES_PATH + "/spynet-estimates-batch" 

    if not os.path.exists(path):
        os.mkdir(path)
      
    imname =  "spynet_frame_" + str(count) + ".png"
    filename = os.path.join(path, imname)
    # filename = "prova_viz.png"
    cv2.imwrite(filename, img_flo)
    print(f"Saved to {filename}")


def demo(args):

    images = glob.glob(os.path.join(args.path, '*.png')) + \
             glob.glob(os.path.join(args.path, '*.jpg'))   

    images = sorted(images)
    count = 0

    for imfile1, imfile2 in zip(images[:-1], images[1:]):

        image1 = load_image(imfile1)
        image2 = load_image(imfile2)

        intHeight = image1.shape[1]
        intWidth = image1.shape[2]
        tenPreprocessedOne = image1.view(1, 3, intHeight, intWidth)
        tenPreprocessedTwo = image2.view(1, 3, intHeight, intWidth)



        flow_out = estimate(tenPreprocessedOne, tenPreprocessedTwo)
        flow_out = flow_out[0,:,:,:]

        viz(image1, flow_out, count)

        if args.out is not None:
            filename = args.out + str(count)
            objOutput = open(filename, 'wb')

            np.array([ 80, 73, 69, 72 ], np.uint8).tofile(objOutput)  
            np.array([ tenOutput.shape[2], tenOutput.shape[1] ], np.int32).tofile(objOutput)
            np.array(tenOutput.numpy().transpose(1, 2, 0), np.float32).tofile(objOutput)

            objOutput.close()

        count +=1


def demo_batch(args):

    image_filenames = glob.glob(os.path.join(args.path, '*.png')) + \
             glob.glob(os.path.join(args.path, '*.jpg'))   

    image_filenames = sorted(image_filenames)

    images = []
    for imfile in image_filenames:
        images.append(load_image(imfile))

    images = torch.stack(images)


    images_batch1 = images[:-1]
    images_batch2 = images[1:]          
    flow_out = estimate(images_batch1, images_batch2) 

    for count, image in enumerate(images[:-1]):
        viz(image, flow_out[count], count)  


        # viz(image1, flow_out, count)

        # if args.out is not None:
        #     filename = args.out + str(count)
        #     objOutput = open(filename, 'wb')

        #     np.array([ 80, 73, 69, 72 ], np.uint8).tofile(objOutput)  
        #     np.array([ tenOutput.shape[2], tenOutput.shape[1] ], np.int32).tofile(objOutput)
        #     np.array(tenOutput.numpy().transpose(1, 2, 0), np.float32).tofile(objOutput)

        #     objOutput.close()

        # count +=1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--out', help="output .flo filename")
    args = parser.parse_args()

    FRAMES_PATH = args.path
    MODEL = args.model

    # t = time.time()
    # demo_batch(args)
    # t_batch = time.time() - t


    t = time.time()
    demo(args)
    t_iter = time.time() - t

    # print(f"Batch elapsed time: {t_batch}")
    print(f"Iter elapsed time: {t_iter}")


    # print("Iterative", timeit.timeit(lambda: demo(args)))


    
# end