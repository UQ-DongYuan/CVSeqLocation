import glob
import random
import os
import json
import math

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

def LatLngToPixel(lat, lng, centerLat, centerLng, zoom):
    # x: height(lat) y: width(lon)
    x, y = LatLngToGlobalPixel(lat, lng, zoom)
    cx, cy = LatLngToGlobalPixel(centerLat, centerLng, zoom)
    return x - cx, y - cy

def LatLngToGlobalPixel(lat, lng, zoom):
    siny = math.sin(lat * math.pi / 180.0)
    siny = min(max(siny, -0.9999), 0.9999)

    return [(256 * (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi))) * (2 ** zoom),
            (256 * (0.5 + lng / 360.0)) * (2 ** zoom)]

def gps2pixel(lat_s, lon_s, lat_d, lon_d, resolution):
    r = 6378137  # equatorial radius
    flatten = 1 / 298257  # flattening
    E2 = flatten * (2 - flatten)
    m = r * np.pi / 180
    lat = (lat_s + lat_d) / 2
    coslat = np.cos(lat * np.pi / 180)
    w2 = 1 / (1 - E2 * (1 - coslat * coslat))
    w = np.sqrt(w2)
    kx = m * w * coslat
    ky = m * w * w2 * (1 - E2)
    x = (lon_d - lon_s) * kx
    y = (lat_s - lat_d) * ky  # y: from top to bottom

    return y / resolution, x / resolution

def input_transform(size):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

class SeqGeoDataset(Dataset):
    def __init__(self, root="/media/dongyuan/DATA/SeqGeo/SeqGeo_dataset", sequence_size=7, mode='train', zoom=20):
        self.zoom = zoom
        self.sequence_size = sequence_size
        self.mode = mode
        self.year = 2019
        self.zoom = zoom
        self.sat_size = [256, 256]
        self.grd_size = [270, 480]
        self.root = root
        self.resolution = 0.114
        self.sat_transform = input_transform(self.sat_size)
        self.grd_transform = input_transform(self.grd_size)

        self.json_files = sorted(glob.glob(os.path.join(root, 'json',  str(self.year) + "_JSON") + '/*.json'),
                                 key=lambda x: int(x.split("/")[-1].split(".json")[0]))
        # split dataset
        if mode == "train":
            self.json_files = self.json_files[:int(len(self.json_files) * 0.8)]
        elif mode == "test":
            self.json_files = self.json_files[int(len(self.json_files) * 0.8 + 1):]
            # self.json_files = self.json_files[:int(len(self.json_files) * 0.8)]

        feature_len = 1024
        grid_size = (32, 32)
        self.stride = self.sat_size[0] / grid_size[0]
        self.grid = torch.arange(0, feature_len, 1)
        self.grid = torch.reshape(self.grid, grid_size)  # 32 x 32

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, index):
        # open and load json file
        f = open(self.json_files[index])
        meta_data = json.load(f)
        sate_center_lat, sate_center_lon = meta_data["center"]
        f.close()

        # load satellite image
        dir_sate_img = meta_data["satellite_views"][str(self.zoom)]
        dir_sate_img = dir_sate_img.split("\\")[1:]
        dir_sate_img = "/".join(dir_sate_img)
        sate_img = self.sat_transform(Image.open(os.path.join(self.root, "satellite", dir_sate_img)).convert('RGB'))

        # load street view images
        all_street_views = meta_data["street_views"]
        # keep all continuous sequence size to 7
        current_len = len(all_street_views.keys())
        if current_len > self.sequence_size:
            for i in range(8, current_len + 1):
                all_street_views.pop(f'img_{i}')
        # print(all_street_views.keys())
        assert len(all_street_views.keys()) == self.sequence_size
        del current_len
        # print(all_street_views.keys())
        street_images = []
        labels = []
        gt_xy = []
        for k in sorted(all_street_views.keys()):
            v = all_street_views[k]
            row_offset, col_offset = gps2pixel(sate_center_lat, sate_center_lon, v["lat"], v["lon"], self.resolution)
            row_offset_resized = int(row_offset * self.sat_size[0] / 640)
            col_offset_resized = int(col_offset * self.sat_size[0] / 640)
            # grd position (Gy, Gx) related to sat_size (256, 256)
            gt_y, gt_x = (self.sat_size[0] / 2) - 1 + row_offset_resized, (
                        self.sat_size[1] / 2) - 1 + col_offset_resized
            # compute grid index
            grid_y, grid_x = int(gt_y // self.stride), int(gt_x // self.stride)
            idx = self.grid[grid_y, grid_x].item()  # [0, 1023]
            ty, tx = gt_y / self.stride - grid_y, gt_x / self.stride - grid_x

            dir_img = v["name"]
            dir_img = dir_img.split("\\")[1:]
            dir_img = "/".join(dir_img)
            dir_img = os.path.join(self.root, "street", os.path.join(str(self.year) + "_street", dir_img))
            img = self.grd_transform(Image.open(dir_img).convert('RGB'))
            img = img[:, 10:, :]

            street_images.append(img)
            labels.append((idx, ty, tx))
            gt_xy.append((gt_y, gt_x))

        # stack to torch tensors on dim=0
        street_images = torch.stack(tuple(street_images), 0)
        return sate_img, street_images, labels, gt_xy

def data_collect(batch):
    sat_imgs = []
    grd_imgs = []
    labels = []
    ground_yx = []
    for sample in batch:
        sat_imgs.append(sample[0])
        grd_imgs.append(sample[1])
        labels.append(sample[2])
        ground_yx.append(sample[3])
    return torch.stack(sat_imgs, 0), torch.stack(grd_imgs, 0), labels, ground_yx

if __name__ == "__main__":
    dataset = SeqGeoDataset(mode='train')
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=data_collect)
    for i, (sat, grd, label, gt_p) in tqdm(enumerate(dataloader), leave=True):
        print(sat.shape)
        print(grd.shape)
        print(label)
        print(gt_p)
        # save_image(sat, 'sat_test.png')
        # save_image(grd, 'grd_test.png')
        break
