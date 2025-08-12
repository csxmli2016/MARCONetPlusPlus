
import torch
from torch.hub import download_url_to_file, get_dir
import cv2 
import numpy as np
import os.path
import torch.nn.functional as F
import time
import argparse
import os.path as osp
from urllib.parse import urlparse
from models.TextEnhancement import MARCONetPlus as MARCONetPlus
from utils.utils_image import get_image_paths, imread_uint, uint2tensor4, tensor2uint
from networks.rrdbnet2_arch import RRDBNet as BSRGAN




def inference(input_path=None, output_path=None, aligned=False, save_text=False, device=None):

    scale_factor = 4 # upsample scale factor for the final output, fixed
    if device == None or device == 'gpu':
        use_cuda = torch.cuda.is_available()
    if device == 'cpu':
        use_cuda = False
    device = torch.device('cuda' if use_cuda else 'cpu')

    if input_path is None:
        exit('input image path is none. Please see our document')

    if output_path is None:
        TIMESTAMP = time.strftime("%m-%d_%H-%M", time.localtime())
        if input_path[-1] == '/' or input_path[-1] == '\\':
            input_path = input_path[:-1]
        output_path = osp.join(input_path+'_'+TIMESTAMP+'_MARCONetPlus')
    os.makedirs(output_path, exist_ok=True)

    # use bsrgan to restore the background of the whole image
    if not aligned: 
        ##BG model
        BGModel = BSRGAN(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=2)  # define network
        model_old = torch.load('./checkpoints/bsrgan_bg.pth')
        state_dict = BGModel.state_dict()
        for ((key, param),(key2, param2)) in zip(model_old.items(), state_dict.items()):
            state_dict[key2] = param
        BGModel.load_state_dict(state_dict, strict=True)
        BGModel.eval()
        for k, v in BGModel.named_parameters():
            v.requires_grad = False
        BGModel = BGModel.to(device)
        torch.cuda.empty_cache()
    

    lq_paths = get_image_paths(input_path)
    if len(lq_paths) ==0:
        exit('No Image in the LR path.')


    WEncoderPath='./checkpoints/net_w_encoder_860000.pth'
    PriorModelPath='./checkpoints/net_prior_860000.pth'
    SRModelPath='./checkpoints/net_sr_860000.pth'

    TextModel = MARCONetPlus(WEncoderPath, PriorModelPath, SRModelPath, device=device)

    print('{:>25s} : {:s}'.format('Model Name', 'MARCONetPlusPlus'))
    if use_cuda:
        print('{:>25s} : {:<d}'.format('GPU ID', torch.cuda.current_device()))
    else:
        print('{:>25s} : {:s}'.format('GPU ID', 'No GPU is available. Use CPU instead.'))
    torch.cuda.empty_cache()

    L_path = input_path
    E_path = output_path # save path
    print('{:>25s} : {:s}'.format('Input Path', L_path))
    print('{:>25s} : {:s}'.format('Output Path', E_path))
    if aligned:
        print('{:>25s} : {:s}'.format('Image Details', 'Aligned Text Layout. No text detection is used.'))
    else:
        print('{:>25s} : {:s}'.format('Image Details', 'UnAligned Text Image. It will crop text region using CnSTD, restore, and paste results back.'))
    print('{:>25s} : {:s}'.format('Save LR & SR text layout', 'True' if save_text else 'False'))

    idx = 0    

    for iix, img_path in enumerate(lq_paths):
        ####################################
        #####(1) Read Image
        ####################################
        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img_path))
        print('{:>20s} {:04d} --> {:<s}'.format('Restoring ', idx, img_name+ext))

        img_L = imread_uint(img_path, n_channels=3) #RGB 0~255
        height_L, width_L = img_L.shape[:2]

        if not aligned:
            img_E = cv2.resize(img_L, (int(width_L//8*8), int(height_L//8*8)), interpolation = cv2.INTER_AREA)
            img_E = uint2tensor4(img_E).to(device) #N*C*W*H 0~1
            with torch.no_grad():
                try:
                    img_E = BGModel(img_E)
                except:

                    del img_E
                    torch.cuda.empty_cache()
                    max_size = 1536
                    print(' ' * 25 + f' ... Background image is too large... OOM... Resize the image with maximum dimension at most {max_size} pixels')
                    scale = min(max_size / width_L, max_size / height_L, 1.0)  
                    new_width = int(width_L * scale)
                    new_height = int(height_L * scale)
                    img_E = cv2.resize(img_L, (new_width//8*8, new_height//8*8), interpolation=cv2.INTER_AREA)
                    img_E = uint2tensor4(img_E).to(device)
                    img_E = BGModel(img_E)

            img_E = tensor2uint(img_E)
        else:
            img_E = img_L

        width_S = (width_L * scale_factor)
        height_S = (height_L * scale_factor)
        img_E = cv2.resize(img_E, (width_S, height_S), interpolation = cv2.INTER_AREA)

        #########################################################
        #####(2) Restore Each Region and Paste to the whole image
        #########################################################
        
        SQ, ori_texts, en_texts, debug_texts, pred_texts = TextModel.handle_texts(img=img_L, bg=img_E, sf=scale_factor, is_aligned=aligned)

        
        ext = '.png'
        if SQ is None:
            continue
        if not aligned:
            SQ = cv2.resize(SQ.astype(np.float32), (width_S, height_S), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(E_path, img_name+ext), SQ[:,:,::-1])
        else:
            cv2.imwrite(os.path.join(E_path, img_name+ext), en_texts[0][:,:,::-1])

        ####################################
        #####(3) Save Cropped Results
        ####################################
        if save_text:
            for m, (et, ot, dt, pt) in enumerate(zip(en_texts, ori_texts, debug_texts, pred_texts)): ##save each face region
                w, h, c = et.shape
                cv2.imwrite(os.path.join(E_path, img_name +'_patch_{}_{}_Debug.jpg'.format(m, pt)), dt[:,:,::-1])

    
if __name__ == '__main__':
    '''
    For the whole image: python test_marconetplus.py -s -i ./testsets/LR
    For the aligned text image: python test_marconetplus.py -a -s -i ./testsets/LR_Crop

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, default='./testsets/LR_Whole', help='The lr text image path')
    parser.add_argument('-o', '--output_path', type=str, default=None, help='The save path for text sr result')
    parser.add_argument('-a', '--aligned', action='store_true', help='The input text image contains only text region or not, default:False')
    parser.add_argument('-s', '--save_text', action='store_true', help='Save the LR, SR and debug text layout or not')
    parser.add_argument('-d', '--device', type=str, default=None, help='using cpu or gpu')

    args = parser.parse_args()
    inference(args.input_path, args.output_path, args.aligned, args.save_text, args.device)


