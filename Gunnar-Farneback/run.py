import argparse
import glob
import os
import numpy as np
import cv2
import flow_viz


FRAMES_PATH = ''
OUT_FOLDER = ''

def load_image(imfile):
    img = cv2.imread(imfile)

    return img
    

def viz(img, flo, count = ""):
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    flo = cv2.cvtColor(flo, cv2.COLOR_RGB2BGR)

    img_flo = np.concatenate([img, flo], axis=0)


    path = FRAMES_PATH + "/" + OUT_FOLDER + "-estimates" 
    if not os.path.exists(path):
        os.mkdir(path)
       
    imname =  OUT_FOLDER + "_frame_" + str(count) + ".png"
    filename = os.path.join(path, imname)
    cv2.imwrite(filename, img_flo)
    print(f"Saved to {filename}")


def compute_flow(args):
    images = glob.glob(os.path.join(args.path, '*.png')) + \
                     glob.glob(os.path.join(args.path, '*.jpg'))
            
    images = sorted(images)
    count = 0
    
    image1 = load_image(images[0])
    for imfile2 in images[1:]:
        image2 = load_image(imfile2)

        prvs = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        nxt = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        if OUT_FOLDER is not None:
            viz(image2, flow, count)

        image1 = image2
        count += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--output_folder', help="name of folder where to save results")

    args = parser.parse_args()

    FRAMES_PATH = args.path
    OUT_FOLDER = args.output_folder

    import time

    t = time.time()
    compute_flow(args)
    print(f"Elapsed time of Gunnar-Farneback algorithm: {time.time() - t}")

