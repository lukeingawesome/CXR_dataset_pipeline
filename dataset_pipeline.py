# Standard library imports
import os
import re
from typing import Tuple

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import cv2
import yaml
import hydra
from tqdm import tqdm

# Local imports
from utils import load_image, get_transform
from oodnet import OODNet

# Constants
CONFIG_PATHS = {
    'mono': './configs/cxr-mono.yaml',
    'rotation': './configs/cxr-rot.yaml',
    'view': './configs/cxr-ood.yaml'
}

MODEL_PATHS = {
    'mono': 'weight/mono.ckpt',
    'rotation': 'weight/rotation.ckpt',
    'view': 'weight/epoch_appa.ckpt',
    'ood1': 'weight/model3.tar',
    'ood2': 'weight/model4.tar'
}

class ImageProcessor:
    """Handles all image processing operations."""
    
    @staticmethod
    def remove_black_padding(image: np.ndarray, margin: int = 0) -> np.ndarray:
        """Removes black padded space from an X-ray image."""
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return image[y+margin:y+h-margin, x+margin:x+w-margin]

    @staticmethod
    def resize_with_aspect_ratio(img: np.ndarray, max_size: int) -> np.ndarray:
        """Resizes image maintaining aspect ratio."""
        if img is None:
            raise ValueError("Image not found or cannot be read")

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        h, w = img.shape
        aspect_ratio = w / h
        
        if w < h:
            new_w = max_size
            new_h = int(new_w / aspect_ratio)
        else:
            new_h = max_size
            new_w = int(new_h * aspect_ratio)
            
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def inverse(img: np.ndarray) -> np.ndarray:
        """Inverts the image."""
        return img.max() - img

class ModelHandler:
    """Handles model loading and inference operations."""
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self, config: dict, ckpt_path: str) -> nn.Module:
        """Loads and returns a model from checkpoint."""
        model_config = config['model']
        model = hydra.utils.instantiate(model_config)
        model = model.to(self.device)
        
        ckpt = torch.load(ckpt_path, map_location=self.device)
        ckpt = {re.sub(r'^model.', '', k): v for k, v in ckpt['state_dict'].items()}
        model.load_state_dict(ckpt)
        return model.eval()

    def load_ood_models(self) -> Tuple[nn.Module, nn.Module, np.ndarray, np.ndarray]:
        """Loads OOD models and their centers."""
        nets = []
        centers = []
        
        for path in [MODEL_PATHS['ood1'], MODEL_PATHS['ood2']]:
            net = OODNet().to(self.device)
            model_dict = torch.load(path, map_location=self.device)
            net.load_state_dict(model_dict['net_dict'])
            nets.append(net.eval())
            centers.append(model_dict['c'])
            
        return nets[0], nets[1], centers[0], centers[1]

def load_raw_img(img_pth: str, resize_factor: int) -> np.ndarray:
    width_param = 4.0
    img = load_image(img_pth, do_windowing=False, width_param=width_param)
    img = img * 255.0
    img = img.astype(np.uint8)
    img = ImageProcessor.remove_black_padding(img)
    img = ImageProcessor.resize_with_aspect_ratio(img, resize_factor)
    return img

def load_img(img_pth: str, transforms, resize_factor: int) -> np.ndarray:
    width_param = 4.0
    img = load_image(img_pth, do_windowing=True, width_param=width_param)
    img = img * 255.0
    img = img.astype(np.uint8)
    img = ImageProcessor.remove_black_padding(img)
    img = ImageProcessor.resize_with_aspect_ratio(img, resize_factor)
    img = transforms(image=img)
    return img['image']

def load_data(img_path: str, device: str, resize_factor: int) -> torch.Tensor:
    image = cv2.imread(img_path, -1)
    image = cv2.resize(image, (resize_factor, resize_factor))
    image = (image - image.mean()) / (image.std() + 1e-8)
    img = torch.from_numpy(image).to(device)
    return img[None,None,:,:]

@torch.no_grad()
def _infer_image(model: torch.nn.Module, img: np.ndarray, mean, std) -> np.ndarray:
    img = F.to_tensor(img)
    img = F.normalize(img, [mean], [std])
    if torch.cuda.is_available():
        img = img.unsqueeze(0)
        img = img.cuda(non_blocking=True)
    return model(img).cpu().numpy()

@torch.no_grad()
def _infer_ood(model: torch.nn.Module, img_pth: str, c):
    img = load_data(img_pth, 'cuda' if torch.cuda.is_available() else 'cpu', 512)
    out = model(img.float())
    out = np.array(out.squeeze().cpu())
    dist = sum((out-c)**2)
    return dist

def postprocess(model, dist):
    o = {}
    scores = model.header.alphas.detach().cpu().numpy() - dist
    pred = softmax(scores)
    preds = np.argmax(scores)
    o['dist'] = dist
    o['score'] = pred
    o['pred'] = preds
    return o

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def main(source: str, dest: str, monos: bool, dataset: str, resize_factor: int) -> None:
    df = pd.DataFrame()
    file_list = os.listdir(source)

    with open(CONFIG_PATHS['mono'], 'r') as f:
        config_mono = yaml.load(f, yaml.FullLoader)
    with open(CONFIG_PATHS['rotation'], 'r') as f:
        config_rot = yaml.load(f, yaml.FullLoader)
    
    transforms = get_transform(config_mono['datamodule']['transforms']['test'])
    mean = config_mono['datamodule']['dataset_config']['mean']
    std = config_mono['datamodule']['dataset_config']['std']

    mono_model = ModelHandler().load_model(config_mono, MODEL_PATHS['mono'])
    rot_model = ModelHandler().load_model(config_rot, MODEL_PATHS['rotation'])

    with open(CONFIG_PATHS['view'], 'r') as f:
        config_view = yaml.load(f, yaml.FullLoader)

    transforms_view = get_transform(config_view['datamodule']['transforms']['test'])
    view_model = ModelHandler().load_model(config_view, MODEL_PATHS['view'])
    ood1, ood2, c1, c2 = ModelHandler().load_ood_models()

    print('Starting : Monochrome and Rotation Model Inference')
    for i in tqdm(range(len(file_list))):
        try:
            img_pth = os.path.join(source, file_list[i])
            img = load_img(img_pth, transforms, resize_factor)
            img_r = load_raw_img(img_pth, resize_factor)
            pred_mono = _infer_image(mono_model, img, mean, std)
        except Exception as e:
            print(f"error: {file_list[i]} - {str(e)}")
            df.loc[i, 'object_id'] = file_list[i][:-4]
            df.loc[i, 'error'] = 1
            continue

        mono = 0 if pred_mono[0] < 0.5 else 1

        if monos and mono == 1:
            img = ImageProcessor.inverse(img)
            img_r = ImageProcessor.inverse(img_r)

        pred_rot = _infer_image(rot_model, img, mean, std)
        rot = np.argmax(pred_rot)
        if rot == 1:
            img_r = np.rot90(img_r, k=3)
        elif rot == 2:
            img_r = np.rot90(img_r, k=2)
        elif rot == 3:
            img_r = np.rot90(img_r, k=1)

        os.makedirs(os.path.join(dest, 'preprocessed'), exist_ok=True)
        preprocessed_dest = os.path.join(os.path.join(dest, 'preprocessed'))
        new_dest = os.path.join(preprocessed_dest, file_list[i])
        
        if not os.path.exists(new_dest):
            cv2.imwrite(new_dest, img_r)
            
        df.loc[i, 'object_id'] = file_list[i][:-4]
        df.loc[i, 'img_path'] = new_dest
        df.loc[i, 'mono_pred'] = mono
        df.loc[i, 'rot_pred'] = rot

    print(f'Saved Files on {preprocessed_dest}')
    print('Starting : View_classification & OOD Model')
    
    for i in tqdm(range(len(df))):
        img_pth = df['img_path'][i]
        try:
            img = load_img(img_pth, transforms_view, 512)
            pred = _infer_image(view_model, img, mean, std)
            pred_dict = postprocess(view_model, pred)
        except:
            df.loc[i, 'error'] = 1
            continue

        dist1 = _infer_ood(ood1, img_pth, c1)
        dist2 = _infer_ood(ood2, img_pth, c2)

        view_pos = pred_dict['pred']
        df.loc[i, 'view_pred'] = 'AP' if view_pos == 0 else ('PA' if view_pos == 1 else 'LAT')
        df.loc[i, 'view_score'] = max(pred_dict['score'][0])
        df.loc[i, 'view_dist'] = min(pred_dict['dist'][0][:2])

        df.loc[i, 'ood_dist1'] = np.round(dist1, 4)
        df.loc[i, 'ood_dist2'] = np.round(dist2, 4)
        df.loc[i, 'ood_dist'] = np.round((dist1+dist2), 4)
        df.loc[i, 'ood_total'] = np.round((dist1+dist2), 4) + (min(pred_dict['dist'][0][:2])*0.2)
    
        if i % 1000 == 0:
            df.to_csv(os.path.join(dest, f'data_result_{dataset}.csv'), index=False)

    df = df.reset_index(drop=True)
    destdest = os.path.join(dest, f'data_result_{dataset}.csv')
    df.to_csv(destdest, index=False)
    print(f"saved result at {destdest}")
    print('Done')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/data3/tmp/practice/original/', help="img_path")
    parser.add_argument('--dest', default='/data3/tmp/practice/preprocessed/', type=str, help="destination for refined image")
    parser.add_argument('--mono', default=True, type=bool, help="Whether to save the monochrome model result")
    parser.add_argument('--dataset', default='sample', type=str, help="dataset name")
    parser.add_argument('--resize_factor', default=512, type=int, help="Image resize factor")
    args = parser.parse_args()

    main(args.source, args.dest, args.mono, args.dataset, args.resize_factor)
