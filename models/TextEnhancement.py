# -*- coding: utf-8 -*-
import cv2
import os.path as osp
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
logging.getLogger('modelscope').disabled = True

from cnstd import CnStd
from utils.utils_transocr import get_alphabet
from utils.yolo_ocr_xloc import get_yolo_ocr_xloc
from ultralytics import YOLO

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from networks import *
import warnings
warnings.filterwarnings('ignore')

from modelscope import snapshot_download


##########################################################################################
###############Text Restoration Model revised by xiaoming li
##########################################################################################

alphabet_path = './models/benchmark_cvpr23.txt'
CommonWordsForOCR = get_alphabet(alphabet_path)
CommonWords = CommonWordsForOCR[2:-1]



def str2idx(text):
    idx = []
    for t in text:
        idx.append(CommonWords.index(t) if t in CommonWords else 3484) #3955
    return idx

def get_parameter_details(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    return num_params / 1e6

def tensor2numpy(tensor):
    tensor = tensor * 0.5 + 0.5
    tensor = tensor.squeeze(0).permute(1, 2, 0).flip(2)
    return np.clip(tensor.float().cpu().numpy(), 0, 1) * 255.0


class MARCONetPlus(object):
    def __init__(self, WEncoderPath=None, PriorModelPath=None, SRModelPath=None, YoloPath=None, device='cuda'):
        self.device = device

        modelscope_dir = snapshot_download('damo/cv_convnextTiny_ocr-recognition-general_damo', cache_dir='./checkpoints/modelscope_ocr')
        self.modelscope_ocr_recognition = pipeline(Tasks.ocr_recognition, model=modelscope_dir)
        self.yolo_character = YOLO(YoloPath)

        self.modelWEncoder = PSPEncoder() # WEncoder()
        self.modelWEncoder.load_state_dict(torch.load(WEncoderPath)['params'], strict=True)
        self.modelWEncoder.eval()
        self.modelWEncoder.to(device)

        self.modelPrior = TextPriorModel()
        self.modelPrior.load_state_dict(torch.load(PriorModelPath)['params'], strict=True)
        self.modelPrior.eval()
        self.modelPrior.to(device)
    
        self.modelSR = SRNet()
        self.modelSR.load_state_dict(torch.load(SRModelPath)['params'], strict=True)
        self.modelSR.eval()
        self.modelSR.to(device)


        print('='*128)
        print('{:>25s} : {:.2f} M Parameters'.format('modelWEncoder', get_parameter_details(self.modelWEncoder)))
        print('{:>25s} : {:.2f} M Parameters'.format('modelPrior', get_parameter_details(self.modelPrior)))
        print('{:>25s} : {:.2f} M Parameters'.format('modelSR', get_parameter_details(self.modelSR)))
        print('='*128)

        torch.cuda.empty_cache()
        self.cnstd = CnStd(model_name='db_resnet34',rotated_bbox=True, model_backend='pytorch', box_score_thresh=0.3, min_box_size=10, context=device)
        self.insize = 32


    def handle_texts(self, img, bg=None, sf=4, is_aligned=False, lq_label=None):
        '''
        Parameters:
            img: RGB 0~255.
        '''

        height, width = img.shape[:2]
        bg_height, bg_width = bg.shape[:2]
        print(' ' * 25 + f' ... The input->output image size is {bg_height//sf}*{bg_width//sf}->{bg_height}*{bg_width}')
        
        full_mask_blur = np.zeros(bg.shape, dtype=np.float32)
        full_mask_noblur = np.zeros(bg.shape, dtype=np.float32)
        full_text_img = np.zeros(bg.shape, dtype=np.float32) #+255

        orig_texts, enhanced_texts, debug_texts, pred_texts = [], [], [], []
        ocr_scores = []

        if not is_aligned:
            box_infos = self.cnstd.detect(img)
            for iix, box_info in enumerate(box_infos['detected_texts']):
                box = box_info['box'].astype(int)# left top, right top, right bottom, left bottom, [width, height]
                score = box_info['score']
                if score < 0.5:
                    continue

                extend_box = box.copy()
                w = int(np.linalg.norm(box[0] - box[1]))
                h = int(np.linalg.norm(box[0] - box[3]))

                # extend the bounding box
                extend_lr = 0.15 * h  
                extend_tb = 0.05 * h 
                vec_w = (box[1] - box[0]) / w
                vec_h = (box[3] - box[0]) / h

                extend_box[0] = box[0] - vec_w * extend_lr - vec_h * extend_tb
                extend_box[1] = box[1] + vec_w * extend_lr - vec_h * extend_tb
                extend_box[2] = box[2] + vec_w * extend_lr + vec_h * extend_tb
                extend_box[3] = box[3] - vec_w * extend_lr + vec_h * extend_tb
                extend_box = extend_box.astype(int)

                w = int(np.linalg.norm(extend_box[0] - extend_box[1]))
                h = int(np.linalg.norm(extend_box[0] - extend_box[3]))
                
                if w > h:
                    ref_h = self.insize
                    ref_w = int(ref_h * w / h)
                else:
                    print(' ' * 25 + ' ... Can not handle vertical text temporarily')
                    continue

                ref_point = np.float32([[0,0], [ref_w, 0], [ref_w, ref_h], [0, ref_h]])
                det_point = np.float32(extend_box)

                matrix = cv2.getPerspectiveTransform(det_point, ref_point)
                inv_matrix = cv2.getPerspectiveTransform(ref_point*sf, det_point*sf)

                cropped_img = cv2.warpPerspective(img, matrix, (ref_w, ref_h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_LINEAR)


                in_img, SQ, save_debug, pred_text, preds_locs_txt = self._process_text_line(cropped_img)
                if in_img is None:
                    continue
                h_crop, w_crop = cropped_img.shape[:2]
                SQ = cv2.resize(SQ, (w_crop * sf, h_crop * sf), interpolation=cv2.INTER_LINEAR)
                
                debug_texts.append(save_debug)
                orig_texts.append(in_img)
                enhanced_texts.append(SQ)
                pred_texts.append(''.join(pred_text))

                tmp_mask = np.ones(SQ.shape).astype(float)
                warp_mask = cv2.warpPerspective(tmp_mask, inv_matrix, (bg_width, bg_height), flags=3)
                warp_img = cv2.warpPerspective(SQ, inv_matrix, (bg_width, bg_height), flags=3)
 

                # erode and blur based on the height of text region
                blur_pad = int(h // 6)

                if blur_pad % 2 == 0:
                    blur_pad += 1
                blur_radius = (blur_pad - 1) // 2
                erode_radius = blur_radius + 1   
                erode_pad = 2 * erode_radius + 1

                kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_pad, erode_pad))
                warp_mask_erode = cv2.erode(warp_mask, kernel_erode, iterations=1)

                # warp_mask_blur = cv2.GaussianBlur(warp_mask_erode, (blur_pad, blur_pad), 0)
                warp_mask_blur = cv2.blur(warp_mask_erode, (blur_pad, blur_pad))

                full_text_img = full_text_img + warp_img
                full_mask_blur = full_mask_blur + warp_mask_blur
                full_mask_noblur = full_mask_noblur + warp_mask
                
                ocr_scores.append(score)


            index = full_mask_noblur > 0
            full_text_img[index] = full_text_img[index]/full_mask_noblur[index]

            full_mask_blur = np.clip(full_mask_blur, 0, 1)
            # fuse the text region back to the background
            final_img = full_text_img * full_mask_blur + bg * (1 - full_mask_blur)


            return final_img, orig_texts, enhanced_texts, debug_texts, pred_texts #, ocr_scores
        
        else: #aligned
            
            in_img, SQ, save_debug, pred_text, preds_locs_txt = self._process_text_line(img)
            if in_img is not None:
                debug_texts.append(save_debug)
                orig_texts.append(in_img)
                enhanced_texts.append(SQ)
                pred_texts.append(''.join(pred_text))
        
        return img, orig_texts, enhanced_texts, debug_texts, pred_texts #, preds_locs_txt

    def _process_text_line(self, img):
        """
        Process a single text line region for text enhancement.
        
        Args:
            img: Input text image
        
        """


        height, width = img.shape[:2]
        if height > width:
            print(' ' * 25 + ' ... Can not handle vertical text temporarily')
            return (None,) * 5
        
        w_norm = int(self.insize * width / height) // 4 * 4
        h_norm = self.insize

        img = cv2.resize(img, (w_norm*4, h_norm*4), interpolation=cv2.INTER_LINEAR)
        in_img = cv2.resize(img, (w_norm, h_norm), interpolation=cv2.INTER_LINEAR)
        ShowLQ = img[:,:,::-1]

        LQ_HeightNorm = transforms.ToTensor()(in_img)
        LQ_HeightNorm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(LQ_HeightNorm).unsqueeze(0).to(self.device)


        '''
        Step 1: Predicting the character labels, bounding boxes.
        '''

        recognized_boxes, pred_text, char_x_centers = get_yolo_ocr_xloc(
            img,                        # input image, RGB 0~255
            yolo_model=self.yolo_character,       # YOLO model instance for character detection
            ocr_pipeline=self.modelscope_ocr_recognition,  # OCR pipeline/model for character recognition
            num_cropped_boxes=5,             # Number of adjacent character boxes to include in each cropped segment (window size)
            expand_px=1,                     # Number of pixels to expand each crop region on all sides (except first/last)
            expand_px_for_first_last_cha=12, # Number of pixels to expand the crop region for the first and last character (left/right respectively)
            yolo_iou=0.1,                    # IOU threshold for YOLO non-max suppression (NMS)
            yolo_conf=0.07                   # Confidence threshold for YOLO detection
        )

        print('{:>25s} ... Recognized chars: {}'.format(' ', ''.join(pred_text)))
        loc_sr = torch.tensor(char_x_centers, device=self.device).unsqueeze(0)

        
        # show character location 
        pad = 1
        ShowPredLoc = ShowLQ.copy()
        for l in range(len(pred_text)):
            center_pred_w = int(loc_sr[0][l].item())
            if center_pred_w > 0: 
                ShowPredLoc[:, max(0, center_pred_w-pad):min(center_pred_w+pad, ShowPredLoc.shape[1]), :] = 0
                ShowPredLoc[:, max(0, center_pred_w-pad):min(center_pred_w+pad, ShowPredLoc.shape[1]), 1] = 255


        '''
        Step 2: Character Prior Generation
        '''

        with torch.no_grad():
            w = self.modelWEncoder(LQ_HeightNorm, loc_sr)
        
        predict_characters128 = []
        predict_characters64 = []
        predict_characters32 = []

        for b in range(w.size(0)):
            w0 = w[b,...].clone() #16*512
            pred_label = str2idx(pred_text)
            pred_label = torch.Tensor(pred_label).type(torch.LongTensor).view(-1, 1)#.to(device)

            with torch.no_grad():
                prior_cha, prior_fea64, prior_fea32 = self.modelPrior(styles=w0[:len(pred_text),:], labels=pred_label, noise=None) #b *n * w * h

            predict_characters128.append(prior_cha)
            predict_characters64.append(prior_fea64)
            predict_characters32.append(prior_fea32)
            
        
        '''
        Step 3: Character SR
        '''

        with torch.no_grad():
            extend_right_width = extend_left_width = h_norm // 2
            LQ_HeightNorm_WidthExtend = F.pad(LQ_HeightNorm, (extend_left_width, extend_right_width, 0, 0), mode='replicate')
            
            preds_locs_txt = ''
            loc_for_extend_sr = loc_sr.clone()
            for i in range(len(pred_text)):
                preds_locs_txt += str(int(loc_for_extend_sr[0][i].cpu().item()))+'_'
                loc_for_extend_sr[0][i] = loc_for_extend_sr[0][i] + extend_left_width * 4
                
            SR = self.modelSR(LQ_HeightNorm_WidthExtend, predict_characters64, predict_characters32, loc_for_extend_sr)
    
        SR = tensor2numpy(SR)[:, extend_left_width * 4:extend_left_width * 4 + w_norm*4, ::-1]

        
        # reduce color inconsistency，use ab channel from in_img
        # sr_lab = cv2.cvtColor(SR.astype(np.uint8), cv2.COLOR_BGR2LAB)
        # target_size = (SR.shape[1], SR.shape[0])
        # in_img_resize = cv2.resize(in_img, target_size, interpolation=cv2.INTER_LINEAR)
        # in_img_lab = cv2.cvtColor(in_img_resize.astype(np.uint8), cv2.COLOR_BGR2LAB)
        # sr_lab[:,:,1:] = in_img_lab[:,:,1:]
        # SR = cv2.cvtColor(sr_lab, cv2.COLOR_LAB2BGR)

        
        prior128 = []
        pad = 2
        for prior in predict_characters128:
            for ii, p in enumerate(prior):
                prior128.append(p)
        prior128 = torch.cat(prior128, dim=2)
        prior128 = prior128 * 0.5 + 0.5
        prior128 = prior128.permute(1, 2, 0).flip(2)
        prior128 = np.clip(prior128.float().cpu().numpy(), 0, 1) * 255.0
        prior128 = np.repeat(prior128, 3, axis=2)

        ShowPrior = cv2.resize(prior128, (SR.shape[1], int(128 * SR.shape[1] / prior128.shape[1])), interpolation=cv2.INTER_LINEAR)
        

        #--------Fuse the structure prior to the LR input to show the details of alignment--------------
        fusion_bg = np.zeros_like(SR, dtype=np.float32)
        w4 = w_norm * 4

        for iii, c in enumerate(loc_sr[0].int()):
            current_prior = prior128[:, iii*128:(iii+1)*128, :]
            center_loc = c.item()

            x1 = max(center_loc - 64, 0)
            x2 = min(center_loc + 64, w4)
            y1 = max(64 - center_loc, 0)
            y2 = y1 + (x2 - x1)
            try:
                fusion_bg[:, x1:x2, :] += current_prior[:, y1:y2, :]
            except:
                return (None,) * 5

        
        mask = fusion_bg / 255.0
        fusion_bg[:,:,0] = 0
        fusion_bg[:,:,2] = 0


        ShowLQ = ShowLQ[:,:,::-1]
        fusion_bg = fusion_bg.astype(ShowLQ.dtype)
        fusion_bg = fusion_bg * 0.3 * mask + ShowLQ * 0.7 * mask + (1-mask) * ShowLQ

        ShowPrior = cv2.normalize(ShowPrior, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        save_debug = np.vstack((ShowLQ, ShowPredLoc[:,:,::-1], SR, ShowPrior, fusion_bg))

        return in_img, SR, save_debug, pred_text, preds_locs_txt



if __name__ == '__main__':
    print('Test')



