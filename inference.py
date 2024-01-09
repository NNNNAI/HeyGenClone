import numpy as np
import cv2, os, sys, subprocess, platform, torch, copy
from tqdm import tqdm
from PIL import Image
from scipy.io import loadmat

sys.path.insert(0, 'third_part')
sys.path.insert(0, 'third_part/GPEN')
sys.path.insert(0, 'third_part/GFPGAN')

# 3dmm extraction
from third_part.face3d.util.preprocess import align_img
from third_part.face3d.util.load_mats import load_lm3d
from third_part.face3d.extract_kp_videos import KeypointExtractor
# face enhancement
from third_part.GPEN.gpen_face_enhancer import FaceEnhancement
from third_part.GFPGAN.gfpgan import GFPGANer
# expression control
from third_part.ganimation_replicate.model.ganimation import GANimationModel

from utils import audio
from utils.ffhq_preprocess import Croper
from utils.alignment_stit import crop_faces, calc_alignment_coefficients, paste_image
from utils.inference_utils import Laplacian_Pyramid_Blending_with_mask, face_detect_with_spec_coords, load_model, options, split_coeff, \
                                  trans_image, transform_semantic, find_crop_norm_ratio, load_face3d_net, exp_aus_dict
import warnings
warnings.filterwarnings("ignore")


def bbox_from_landmarks(lm):
    x1 = np.min(lm[:, 0])
    x2 = np.max(lm[:, 0])
    y1 = np.min(lm[:, 1])
    y2 = np.max(lm[:, 1])
    return x1, x2, y1, y2

# frames:256x256, full_frames: original size
def datagen(frames, mels, full_frames, frames_pil, crop_list, quad_list, frame_id_list, coords_list,base_name,LNet_batch_size,face_det_batch_size,img_size=384):
    img_batch, mel_batch, frame_batch, coords_batch, ref_batch, full_frame_batch, frame_id_batch = [], [], [], [], [], [], []
    refs = []
    image_size = 256 

    # original frames
    kp_extractor = KeypointExtractor()
    fr_pil = [Image.fromarray(frame) for frame in frames]
    lms = kp_extractor.extract_keypoint(fr_pil, 'temp/'+base_name+'x12_landmarks.txt')
    frames_pil = [ (lm, frame) for frame,lm in zip(fr_pil, lms)] # frames is the croped version of modified face
    crops, orig_images, quads  = crop_faces(image_size, frames_pil, scale=1.0, use_fa=True)
    inverse_transforms = [calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]]) for quad in quads]
    del kp_extractor.detector

    full_frames_BGR = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in full_frames]
    # oy1,oy2,ox1,ox2 = cox
    face_det_results = face_detect_with_spec_coords(full_frames_BGR, coords_list, face_det_batch_size, jaw_correction=True)

    for inverse_transform, crop, full_frame, face_det, crop_coord, quad_coord in zip(inverse_transforms, crops, full_frames_BGR, face_det_results, crop_list, quad_list):
        
        clx, cly, crx, cry = crop_coord
        lx, ly, rx, ry = quad_coord
        lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
        oy1, oy2, ox1, ox2 = cly+ly, min(cly+ry, full_frame.shape[0]), clx+lx, min(clx+rx, full_frame.shape[1])
    
    
        imc_pil = paste_image(inverse_transform, crop, Image.fromarray(
            cv2.resize(full_frame[int(oy1):int(oy2), int(ox1):int(ox2)], (256, 256))))

        ff = full_frame.copy()
        ff[int(oy1):int(oy2), int(ox1):int(ox2)] = cv2.resize(np.array(imc_pil.convert('RGB')), (ox2 - ox1, oy2 - oy1))
        oface, coords = face_det
        y1, y2, x1, x2 = coords
        refs.append(ff[y1: y2, x1:x2])

    for i, m in enumerate(mels):
        idx =  i % len(frames)
        frame_to_save = frames[idx].copy()
        face = refs[idx]
        oface, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (img_size, img_size))
        oface = cv2.resize(oface, (img_size, img_size))

        img_batch.append(oface)
        ref_batch.append(face) 
        mel_batch.append(m)
        coords_batch.append(coords)
        frame_id_batch.append(frame_id_list[idx])
        frame_batch.append(frame_to_save)
        full_frame_batch.append(full_frames_BGR[idx].copy())

        if len(img_batch) >= LNet_batch_size:
            img_batch, mel_batch, ref_batch = np.asarray(img_batch), np.asarray(mel_batch), np.asarray(ref_batch)
            img_masked = img_batch.copy()
            img_original = img_batch.copy()
            img_masked[:, img_size//2:] = 0
            img_batch = np.concatenate((img_masked, ref_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch, frame_id_batch
            img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch, ref_batch, frame_id_batch  = [], [], [], [], [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch, ref_batch = np.asarray(img_batch), np.asarray(mel_batch), np.asarray(ref_batch)
        img_masked = img_batch.copy()
        img_original = img_batch.copy()
        img_masked[:, img_size//2:] = 0
        img_batch = np.concatenate((img_masked, ref_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        yield img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch, frame_id_batch
        
        

class VideoRetalking:
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.enhancer = FaceEnhancement(base_dir='checkpoints', size=512, model='GPEN-BFR-512', use_sr=False, \
                               sr_model='rrdb_realesrnet_psnr', channel_multiplier=2, narrow=1, device=device)
        self.restorer = GFPGANer(model_path='checkpoints/GFPGANv1.3.pth', upscale=1, arch='clean', \
                        channel_multiplier=2, bg_upsampler=None)
        self.croper = Croper('checkpoints/shape_predictor_68_face_landmarks.dat')
        
        # self.net_recon = load_face3d_net(args.face3d_net_path, device)
        self.net_recon = load_face3d_net("checkpoints/face3d_pretrain_epoch_20.pth", device)
        self.lm3d_std = load_lm3d('checkpoints/BFM')
        os.makedirs('temp', exist_ok=True)
        self.DNet_path = 'checkpoints/DNet.pt'
        self.LNet_path = 'checkpoints/LNet.pth'
        self.ENet_path = 'checkpoints/ENet.pth'
        
    def sync(self,frames_dict,audio_file,fps,LNet_batch_size, face_det_batch_size, base_name,re_preprocess=False):
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        wav = audio.load_wav(audio_file, 16000)
        mel = audio.melspectrogram(wav)
        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        mel_step_size, mel_idx_multiplier, i, mels = 16, 80./fps, 0, []
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mels.append(mel[:, len(mel[0]) - mel_step_size:])
                break
            mels.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1
        frame_list = []
        frame_id_list = []
        coords_list = []
        mel_chunks = []
        for i in range((len(mels))):
            idx = i % len(frames_dict)
            frame_to_save = frames_dict[idx]['frame'].copy()
            # frame_to_save 是RGB的
            if frames_dict[idx]['has_face']:
                coords = frames_dict[idx]['bbox']
                coords_list.append(coords)
                frame_list.append(frame_to_save)
                frame_id_list.append(idx)
                mel_chunks.append(mels[i])
        frame_to_process = frame_list.copy()
        print('[Lip sync Step 0] cropping each frame into ffhq style.')
        full_frames_RGB_ffhq, crop_list, quad_list = self.croper.crop_eachframe(frame_to_process,coords_list, xsize=512)
        
        frames_pil = [Image.fromarray(cv2.resize(frame,(256,256))) for frame in full_frames_RGB_ffhq]
        print('[Lip sync Step 1] Landmarks Extraction in Video.')
        kp_extractor = KeypointExtractor()
        lm = kp_extractor.extract_keypoint(frames_pil, './temp/'+base_name+'_landmarks.txt')
        
        if not os.path.isfile('temp/'+base_name+'_coeffs.npy'):
            video_coeffs = []
            for idx in tqdm(range(len(frames_pil)), desc="[Lip sync Step 2] 3DMM Extraction In Video:"):
                frame = frames_pil[idx]
                W, H = frame.size
                lm_idx = lm[idx].reshape([-1, 2])
                if np.mean(lm_idx) == -1:
                    lm_idx = (self.lm3d_std[:, :2]+1) / 2.
                    lm_idx = np.concatenate([lm_idx[:, :1] * W, lm_idx[:, 1:2] * H], 1)
                else:
                    lm_idx[:, -1] = H - 1 - lm_idx[:, -1]

                trans_params, im_idx, lm_idx, _ = align_img(frame, lm_idx, self.lm3d_std)
                trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
                im_idx_tensor = torch.tensor(np.array(im_idx)/255., dtype=torch.float32).permute(2, 0, 1).to(device).unsqueeze(0) 
                with torch.no_grad():
                    coeffs = split_coeff(self.net_recon(im_idx_tensor))

                pred_coeff = {key:coeffs[key].cpu().numpy() for key in coeffs}
                pred_coeff = np.concatenate([pred_coeff['id'], pred_coeff['exp'], pred_coeff['tex'], pred_coeff['angle'],\
                                            pred_coeff['gamma'], pred_coeff['trans'], trans_params[None]], 1)
                video_coeffs.append(pred_coeff)
            semantic_npy = np.array(video_coeffs)[:,0]
            np.save('temp/'+base_name+'_coeffs.npy', semantic_npy)    
        else:
            print('[Lip sync Step 2] Using saved coeffs.')
            semantic_npy = np.load('temp/'+base_name+'_coeffs.npy').astype(np.float32)
        # use natural face for the stablize image gen
        expression = torch.tensor(loadmat('checkpoints/expression.mat')['expression_center'])[0]
        
        
        D_Net, model = load_model(self.DNet_path,self.LNet_path,self.ENet_path, device)

        if not os.path.isfile('temp/'+base_name+'_stablized.npy') or re_preprocess:
            imgs = []
            for idx in tqdm(range(len(frames_pil)), desc="[Step 3] Stabilize the expression In Video:"):

                source_img = trans_image(frames_pil[idx]).unsqueeze(0).to(device)
                semantic_source_numpy = semantic_npy[idx:idx+1]
                
                ratio = find_crop_norm_ratio(semantic_source_numpy, semantic_npy)
                coeff = transform_semantic(semantic_npy, idx, ratio).unsqueeze(0).to(device)
            
                # hacking the new expression
                coeff[:, :64, :] = expression[None, :64, None].to(device) 
                with torch.no_grad():
                    output = D_Net(source_img, coeff)
                img_stablized = np.uint8((output['fake_image'].squeeze(0).permute(1,2,0).cpu().clamp_(-1, 1).numpy() + 1 )/2. * 255)
                imgs.append(cv2.cvtColor(img_stablized,cv2.COLOR_RGB2BGR)) 
            np.save('temp/'+base_name+'_stablized.npy',imgs)
            del D_Net
        else:
            print('[Lip sync Step 3] Using saved stabilized video.')
            imgs = np.load('temp/'+base_name+'_stablized.npy')
        torch.cuda.empty_cache()
        
        imgs_enhanced = []
        for idx in tqdm(range(len(imgs)), desc='[Lip sync Step 4] Reference Enhancement'):
            img = imgs[idx]
            # img 结果在"test1.jpg"
            pred, _, _ = self.enhancer.process(img, img, face_enhance=True, possion_blending=False)
            # pred 结果在 "test1_enhance.jpg"
            imgs_enhanced.append(pred)
        full_frames = frame_list.copy()
        gen = datagen(imgs_enhanced.copy(), mel_chunks, full_frames, None, crop_list, quad_list, \
            frame_id_list, coords_list, base_name, LNet_batch_size,face_det_batch_size)
        
        for i, (img_batch, mel_batch, frames, coords, img_original, f_frames, frame_ids) in enumerate(tqdm(gen, desc='[Lip sync Step 5] Lip Synthesis:', total=int(np.ceil(float(len(mel_chunks)) / LNet_batch_size)))):
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
            img_original = torch.FloatTensor(np.transpose(img_original, (0, 3, 1, 2))).to(device)/255. # BGR -> RGB
            
            with torch.no_grad():
                incomplete, reference = torch.split(img_batch, 3, dim=1) 
                pred, low_res = model(mel_batch, img_batch, reference)
                pred = torch.clamp(pred, 0, 1)
                
            
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            torch.cuda.empty_cache()
            for p, f, xf, c, f_id in zip(pred, frames, f_frames, coords, frame_ids):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                
                ff = xf.copy() 
                ff[y1:y2, x1:x2] = p
                
                # month region enhancement by GFPGAN
                cropped_faces, restored_faces, restored_img = self.restorer.enhance(
                    ff, has_aligned=False, only_center_face=True, paste_back=True)
                    # 0,   1,   2,   3,   4,   5,   6,   7,   8,  9, 10,  11,  12,
                mm = [0,   0,   0,   0,   0,   0,   0,   0,   0,  0, 255, 255, 255, 0, 0, 0, 0, 0, 0]
                mouse_mask = np.zeros_like(restored_img)
                tmp_mask = self.enhancer.faceparser.process(restored_img[y1:y2, x1:x2], mm)[0]
                mouse_mask[y1:y2, x1:x2]= cv2.resize(tmp_mask, (x2 - x1, y2 - y1))[:, :, np.newaxis] / 255.

                height, width = ff.shape[:2]
                restored_img, ff, full_mask = [cv2.resize(x, (512, 512)) for x in (restored_img, ff, np.float32(mouse_mask))]
                img = Laplacian_Pyramid_Blending_with_mask(restored_img, ff, full_mask[:, :, 0], 10)
                pp = np.uint8(cv2.resize(np.clip(img, 0 ,255), (width, height)))

                pp, orig_faces, enhanced_faces = self.enhancer.process(pp, xf, bbox=c, face_enhance=False, possion_blending=True)
                
                frames_dict[f_id]['frame'] = cv2.cvtColor(pp, cv2.COLOR_BGR2RGB)
                
        return frames_dict
            
