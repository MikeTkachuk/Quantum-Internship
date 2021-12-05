import os
import numpy as np
import yaml
import pandas as pd
import geopandas as gpd
from PIL import Image
import cv2
import imageio
import math
import matplotlib.pyplot as plt

import rasterio
import rasterio.mask as riomask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.plot import reshape_as_raster, reshape_as_image
from rasterio.profiles import DefaultGTiffProfile
from shapely.ops import cascaded_union
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
from torch.utils.data import Dataset
import albumentations as A
from shapely.geometry import Polygon, shape, LinearRing
from shapely.geometry.collection import GeometryCollection
import skimage.measure as measure
import skimage.morphology as morph

from .models import *

train_add_targets = {'image': 'image',
                     'gt_mask': 'mask',
                     'filter_mask': 'mask'}

train_tfs = A.Compose([A.Transpose(p=0.5),
                       A.HorizontalFlip(p=0.5),
                       A.VerticalFlip(p=0.5),
                       A.ShiftScaleRotate(p=0.5),
                       A.Rotate(p=0.5),
                       ],
                      additional_targets=train_add_targets)

val_tfs = A.Compose([],
                    additional_targets=train_add_targets)

drivers = {
    'tif': 'GTiff',
    'jp2': 'JP2OpenJPEG'
}

def blur_img(img, kernel_size=3):
    img = cv2.medianBlur(img, kernel_size)
    return img
    
    
def preprocess_tile(tile):
    q_stat = np.reshape(tile,(4,1,-1))
    q_stat = q_stat[:,:,q_stat[0,0,:]!=0]
    qs = [np.quantile(q_stat,0.02,axis=-1,keepdims=True),
          np.quantile(q_stat,0.98,axis=-1,keepdims=True)]
    tile = (tile-qs[0])/(qs[1]-qs[0])*255.
    tile = np.clip(tile,0,255).astype(np.float32)
    tile = tile[[3,0,1,2]]
    return tile

    
    
class PolyDataset(Dataset):
    """Dataset object is aimed to prepare the 
    binary masks and images for future training.
    Creates two folders with '_mask' and '_tiles'
    suffixes named by corresponding width and height
    @param: tiles_dir - path to the directory where loaded
                        tile images are stored
    @param: polygons_path - path to the file containing
                            polygons (as of now was tested
                            with geojson where all polygons
                            are combined)
    @param: root_path - a path prefix treated as a root path to the dataset
    @param: dataset_name - a name of the dataset folder to be created or read from
    @param: mini_tile_size - size of images to cut images into,
                                by default images will be cut
                                into 256x256 tiles
    @param: src_format - format of an input image
    @param: bands - list of strings indicating band names present
                    at the end of the filenames. The order affects concatenation
    @param: dst_format - format of output tiles
    @param: dst_crs - CRS of desired format to convert shapes to
    @param: skip_empty_masks - controls the behavior of saving
                                tiles, i.e. if True algorythm will 
                                not write to files images and masks
                                where there is no mask pixels
    @param: transforms - transforms applied to images
    @param: sentinel_scale - whether or not to rescale the data 3,33 times smaller
    @param: preproc_func - optional, func to preprocess tile with. tile -> tile signature.
    """

    def __init__(self, tiles_dir, 
        polygons_path,
        root_path='',
        dataset_name='dataset',
        mask_type='bounds',
        mini_tile_size=256, 
        src_format='tif',
        bands=[],
        dst_format='npy', 
        dst_crs='EPSG:32636',
        skip_empty_masks=True,
        transforms=train_tfs,
        preproc_func=preprocess_tile,
        sentinel_scale=False,
        tensor_type='tf'):

        super().__init__()

        self.polygons_path = polygons_path
        self.root_path = root_path
        self.dataset_name = dataset_name
        self.mask_type=mask_type
        self.mini_tile_size = mini_tile_size
        self.src_format = src_format
        self.dst_crs = rasterio.crs.CRS.from_string(dst_crs)
        self.transforms = transforms
        self.tensor_type = tensor_type
        self.sentinel_scale = sentinel_scale
        self.preproc_func = preproc_func
        # handle band files
        if len(bands):
            band_grouped = dict(zip(bands,[list() for i in range(len(bands))]))
            for file in os.listdir(os.path.join(self.root_path,tiles_dir)):
                for band in bands:
                    if file.endswith(band+'.'+src_format):
                        band_grouped[band].append(os.path.join(self.root_path,tiles_dir,file))
                        break

            self.tile_files = list(zip(*[sorted(band_grouped[x]) for x in bands]))
        else:
            self.tile_files = [(os.path.join(self.root_path,tiles_dir, x),) \
                               for x in os.listdir(os.path.join(self.root_path, tiles_dir)) \
                               if x.endswith('.'+src_format)]
        
        
        self.skip_empty_masks = skip_empty_masks
        
        self._index_data_v2()
        self._prepare_mini_tiles()

        mask_dir = os.path.join(self.root_path, self.dataset_name,self.polygons_path+"_mask")
        tile_dir = os.path.join(self.root_path, self.dataset_name,self.polygons_path+"_tiles")
        filter_dir = os.path.join(self.root_path, self.dataset_name,self.polygons_path+"_filters")
        self.masks = sorted([os.path.join(mask_dir, x) for x in os.listdir(
            mask_dir) if x.endswith('.'+dst_format)])
        self.images = sorted([os.path.join(tile_dir, x) for x in os.listdir(
            tile_dir) if x.endswith('.'+dst_format)])
        self.filters = sorted([os.path.join(filter_dir, x) for x in os.listdir(
            filter_dir) if x.endswith('.'+dst_format)])
        
        
    def _index_data(self):
        poly_list = [x for x in os.listdir(self.polygons_path) 
                     if not x.startswith('.')]
        
        self.polygons_list = sorted([os.path.join(
            self.polygons_path, x) for x in poly_list
                   if not x.endswith('_aoi.geojson')])
        
        self.aois_list = sorted([os.path.join(
            self.polygons_path, x) for x in poly_list
                   if x.endswith('_aoi.geojson')])

        assert len(self.polygons_list) == len(self.aois_list)
        
    def _index_data_v2(self):
        poly_list = [x for x in os.listdir(os.path.join(self.root_path,self.polygons_path)) 
                     if not x.startswith('.')]
        
        self.aois_list = sorted([os.path.join( self.root_path,
            self.polygons_path, x) for x in poly_list
                   if x.endswith('_aoi.geojson')])

        self.polygons_list = [os.path.join( self.root_path,
            self.polygons_path, 'all_locs.geojson')]
        
    def _make_aoi_filter(self, polygons, aoi, crs, save_path=None):

        if isinstance(polygons, list):
            polygons = gpd.GeoDataFrame({'geometry': polygons})
            polygons.crs = crs

        aoi = aoi.to_crs(polygons.crs)
        try:
            chunk = polygons[polygons.within(aoi.geometry.values[0].buffer(50))]
            filters = aoi.difference(chunk.buffer(20).unary_union)
        except Exception as e:
            print(e)

        if save_path is not None:
            filters.to_file(save_path, driver='GeoJSON')

        return filters
        
    def _load_polygons(self, poly_path, dst_crs,
                       cascade=True, aoi=None,
                       return_original=True, borders_width=8):
        """Load shapes to create masks from"""

        polygons = gpd.read_file(poly_path)
        #print(polygons.geometry, polygons.crs, polygons.is_empty)
        try:
            if polygons.crs != dst_crs:
                polygons = polygons.to_crs(dst_crs)
        except Exception as e:
            print(f'Invalid geometry captured, trying to handle: {poly_path}')
            polygons['geometry'] = polygons.geometry.apply(lambda x: x if x else GeometryCollection())
        polygons = polygons[polygons.geometry.is_valid]
        if aoi is not None:
            polygons = polygons[polygons.geometry.intersects(aoi)]
        bounds = polygons.boundary.buffer(borders_width)
        bounds = bounds.geometry.tolist()
        polygons = polygons.geometry.tolist()
        if len(bounds)<1:
            print(f'Warning: {poly_path} as no valid geometries, crs: {polygons.crs}')
        if cascade:
            return [cascaded_union(bounds)], [cascaded_union(polygons)]
        return bounds, polygons

    def _prepare_mini_tiles(self,external_buffer=5):
        """Prepare folders with mini-tiles if folders are abesnt"""

        mask_dir = os.path.join(self.root_path, self.dataset_name,self.polygons_path+"_mask")
        tile_dir = os.path.join(self.root_path, self.dataset_name,self.polygons_path+"_tiles")
        filter_dir = os.path.join(self.root_path, self.dataset_name,self.polygons_path+"_filters")


        if not os.path.exists(mask_dir) or len(list(os.listdir(mask_dir)))==0:
            os.system(f'mkdir -p {mask_dir}')
            os.system(f'mkdir -p {tile_dir}')
            os.system(f'mkdir -p {filter_dir}')
            
            print('Creating folder structure...')
            f,axes = plt.subplots(len(self.tile_files)*3,1,figsize=(10,60))
            c=0
            self.count = 0
            while len(self.tile_files) > 0:

                tile_paths = self.tile_files.pop(0)
                """to_concat = []
                for file in tile_paths:
                    tile = rasterio.open(file) #self._load_tile(file)
                    tile_arr = tile.read()
                    to_concat.append(tile_arr)
                
                tile_arr = np.concatenate(to_concat,axis=0)
                """
                readers = []
                for file in tile_paths:
                    readers.append(rasterio.open(file))
                    
                self.dst_crs = readers[0].crs
                
                for i in range(len(self.aois_list)):
                    aoi = gpd.read_file(self.aois_list[i]).geometry
                    aoi = aoi.to_crs(self.dst_crs)
                    
                    try:
                        regions, transforms = [],[]
                        for reader in readers:
                            region, region_tfs = riomask.mask(
                                reader, aoi, all_touched=False, crop=True)
                            regions.append(region)
                            transforms.append(region_tfs)
                        
                        region = np.concatenate(regions,axis=0)
                        region = self.preproc_func(region)
                        
                        assert all([transforms[i]==transforms[0] for i in range(len(transforms))])
                        regions_tfs = transforms[0]
                        
                    except Exception as e:
                        print(e, f'{tile_paths} and {self.aois_list[i]}')
                        continue
                    bounds, polys = self._load_polygons(
                        self.polygons_list[0], self.dst_crs,
                        cascade=False,
                        aoi=aoi.values[0])
                    
                    fill_bounds = aoi.boundary.buffer(external_buffer)
                    
                    aoi_mask, mask_bounds = self._preprare_mask(
                        fill_bounds, bounds, region_tfs, region.shape)
                    
                    filters = self._make_aoi_filter(
                            polys, aoi, self.dst_crs)
                    
                    mask_filters = rasterio.features.rasterize(
                        shapes=filters,
                        out_shape=(region.shape[-2], region.shape[-1]),
                        transform=region_tfs,
                        default_value=255)              
                    if self.sentinel_scale:
                        # resizing to fit sentinel distribution
                        new_shape = (int(region.shape[2]/3.33),
                                     int(region.shape[1]/3.33))
                        print(region.shape,mask_bounds.shape,mask_filters.shape)
                        region = np.array(
                            Image.fromarray(
                                region.transpose((1,2,0)).astype(np.uint8)
                            ).resize(new_shape)
                        ).transpose(2,0,1)
                        
                        mask_bounds = np.array(
                            Image.fromarray(
                                mask_bounds.astype(np.uint8)
                            ).resize(new_shape)
                        )
                        
                        mask_filters = np.array(
                            Image.fromarray(
                                mask_filters.astype(np.uint8)
                            ).resize(new_shape)
                        )
                    
                    if c<6:
                        axes[c].imshow(np.transpose(region.astype(np.int32),(1,2,0))[:,:,1:])
                        c+=1
                        axes[c].imshow(mask_bounds)
                        c+=1
                        axes[c].imshow(mask_filters)
                        c+=1
                    else:
                        plt.show()
                    self._crop_arrays(
                        region, mask_bounds, mask_filters, 
                        tile_dir, 
                        mask_dir, 
                        filter_dir,
                        tile_paths[0].split('.')[0].split(r'/')[-1])


                for reader in readers:
                    reader.close()

        else:
            print('Dataset has already been created, skipping')
    
    
    def _preprare_mask(self, fill_bounds, bounds, region_tfs, shape):

        aoi_mask = rasterio.features.rasterize(
            shapes=fill_bounds.geometry.tolist(),
            out_shape=(shape[-2], shape[-1]),
            transform=region_tfs,
            default_value=255)

        mask_bounds = rasterio.features.rasterize(
            shapes=bounds,
            out_shape=(shape[-2], shape[-1]),
            transform=region_tfs,
            default_value=255)

        mask_bounds[aoi_mask != 0] = 255

        return aoi_mask, mask_bounds
    
    
    def _crop_arrays(self, tile_arr, 
                     mask, filter_mask, tile_dir='tiles', 
                     mask_dir='mask', filter_dir='filter',
                     tile_name='img',
                     mask_erosions=0):

        for i in tqdm(range(tile_arr.shape[-1]//self.mini_tile_size)):
            for j in range(tile_arr.shape[-2]//self.mini_tile_size):

                width, heigth = int(i*self.mini_tile_size), int(j*self.mini_tile_size)

                #mask_arr = mask[width:width+self.mini_tile_size,
                #    heigth:heigth+self.mini_tile_size]
                #tile_arr_ = tile_arr[:, width:width+self.mini_tile_size,
                #    heigth:heigth+self.mini_tile_size]
                
                mask_arr = mask[heigth:heigth+self.mini_tile_size,
                    width:width+self.mini_tile_size]
                tile_arr_ = tile_arr[:, heigth:heigth+self.mini_tile_size,
                    width:width+self.mini_tile_size]
                
                filters_arr = filter_mask[heigth:heigth+self.mini_tile_size,
                                          width:width+self.mini_tile_size]

                if self.skip_empty_masks and (mask_arr.sum() < 8e4 or tile_arr_.sum() < 8e3):
                    continue
                if mask_erosions !=0:
                    kernel = np.ones((5,5), np.uint8)
                    mask_arr = cv2.erode(mask_arr, kernel, int(mask_erosions))
                
                np.save(
                    os.path.join(mask_dir, f'{tile_name}_{i}_{j}_{self.count}.npy'),
                    np.uint8(mask_arr))
                
                np.save(
                    os.path.join(tile_dir,f'{tile_name}_{i}_{j}_{self.count}.npy'),
                    np.uint8(tile_arr_))
                
                np.save(
                    os.path.join(filter_dir,f'{tile_name}_{i}_{j}_{self.count}.npy'),
                    np.uint8(filters_arr))
                
                self.count += 1

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        
        img = np.load(self.images[idx],allow_pickle=True)
        mask = np.load(self.masks[idx],allow_pickle=True)
        filter_ = np.load(self.filters[idx],allow_pickle=True)

        img = np.transpose(img,(1,2,0))

        if self.transforms is not None:
            transformed = self.transforms(image=img, gt_mask=mask,filter_mask=filter_)
            img = transformed['image']
            mask = transformed['gt_mask']
            filter_ = transformed['filter_mask']
        
        filter_ = np.where(filter_ == 255, True, False)
        img = np.transpose(img,(2,0,1))

        return torch.Tensor(img/255).float(), torch.Tensor(mask/255).float(), torch.Tensor(filter_).bool()
    
    

class BoundaryDetector(object):

    def __init__(self, model,
                 tiles_dir='test', 
                 mask_type='bounds',
                 mini_tile_size=256, 
                 dst_crs='EPSG:4326',
                 src_format='jp2',
                 dst_format='tif'
                ):
        
        self.mask_type=mask_type
        self.mini_tile_size = mini_tile_size
        self.dst_crs = dst_crs
        self.model = model.to(self._get_device())

        self.polygons = []
        self.tile_files = [os.path.join(tiles_dir, x) for x in os.listdir(
            tiles_dir) if x.endswith('.'+src_format)]
        
    def _get_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
        
    def _load_polygons(self, poly_path, dst_crs, cascade=True, return_original=True):
        """Load shapes to create masks from"""

        polygons = gpd.read_file(poly_path)
        try:
            if polygons.crs != dst_crs:
                polygons = polygons.to_crs(dst_crs)
        except Exception as e:
            print(f'Invalid geometry captured, trying to handle: {poly_path}')
            polygons['geometry'] = polygons.geometry.apply(lambda x: x if x else GeometryCollection())
        polygons = polygons[polygons.geometry.is_valid]

        bounds = polygons.geometry.buffer(27) #.difference(polygons.geometry)
        polygons = polygons.geometry.buffer(-7).tolist()
        bounds = bounds.geometry.tolist()
        if len(bounds)<1:
            print(f'Warning: {poly_path} as no valid geometries, crs: {polygons.crs}')
        if cascade:
            return [cascaded_union(bounds)], [cascaded_union(polygons)]
        return bounds, polygons
    
    def _load_tile(self, tile_path):
        with rasterio.open(tile_path) as src:
            image = src.read()
            crs = src.crs
            meta = src.meta
        return image, crs, meta
    
    def _write_raster(self, image, img_path, meta):
        
        if len(image.shape) == 2:
            bands = 1
        else:
            bands = image.shape[0]
            
        with rasterio.open(img_path, 'w', **meta) as dst:
            if bands==1:
                dst.write(image.astype(meta['dtype']), 1)  
                return
            
            for band in range(bands):
                dst.write(image[band].astype(meta['dtype']), band+1)      
        
    def _predict_chip(self, chip, device, conf_thresh=0.5, transforms=val_tfs):
        
        if transforms is not None:
            chip = transforms(image=chip.transpose(1,2,0))['image']
        pred = chip.unsqueeze(0).float()
        
        with torch.no_grad():
            pred = self.model(pred.to(device))
                    
        pred = pred[0][0].cpu().detach().numpy()
        
        return pred
    
    def _get_aoi(self, aoi_path, src_image, meta, dst_crs):
        if aoi_path is not None:
            aoi = gpd.read_file(aoi_path).geometry
            aoi = aoi.to_crs(dst_crs)
                
            tile_arr, region_tfs = riomask.mask(
                src_image, aoi, all_touched=False, crop=True)
            meta['transform'] = region_tfs
            meta['height'] = tile_arr.shape[-2]
            meta['width'] = tile_arr.shape[-1]
            
            aoi_bounds = rasterio.features.rasterize(
                        shapes=aoi.boundary.buffer(20).geometry.tolist(),
                        out_shape=(tile_arr.shape[-2], tile_arr.shape[-1]),
                        transform=region_tfs,
                        default_value=255)
            
        else:
            tile_arr = src_image.read()
            aoi_bounds = cv2.copyMakeBorder(
                np.zeros(tile_arr.shape), 
                10, 10, 10, 10, 
                cv2.BORDER_CONSTANT, 
                value=255)
            
        return aoi_bounds, tile_arr, meta
    
    def _filter_polygons(self, polygons_list, 
                         aoi=None, src_crs='EPSG:4326', 
                         dst_crs='EPSG:4326', 
                         min_poly_area=3e-6):
        
        if aoi is not None:
            aoi = gpd.read_file(aoi)
            aoi = aoi.to_crs(dst_crs)
            aoi_ext = aoi.geometry.values[0].buffer(0.01)
            aoi_ext = LinearRing(aoi_ext.exterior.coords)
        
        df = gpd.GeoDataFrame({"geometry": polygons_list}, 
                              crs=src_crs)
        df = df.to_crs(dst_crs)
        df = df[df.area!=df.area.max()] # it catches exterior of aoi as well :/
        
        df.geometry = df.geometry.buffer(0)
        if aoi is not None:
            df = df[~df.geometry.intersects(aoi_ext)]

        df = df[df.area>min_poly_area]
        df = df.to_crs(dst_crs)
            
        return df
    
    def raster_prediction(self, 
                          in_raster_path, 
                          out_raster_path=None,
                          aoi_path=None,
                          transforms=val_tfs,
                          pred_window=256,
                          step=224,
                          raster_format='tif',
                          conf_thresh=0.5):
        
        device = self._get_device()
        image, crs, meta = self._load_tile(in_raster_path)
        
        if out_raster_path is None:
            src_format = in_raster_path.split('.')[-1]
            out_raster_path = in_raster_path.replace('.', '_prediction.')
            out_raster_path = out_raster_path.replace(src_format, raster_format)
            if in_raster_path.split('.')[-1] != raster_format:
                meta['driver'] = drivers[raster_format]
            
        with rasterio.open(in_raster_path) as src:
            aoi_bounds, image, meta = self._get_aoi(
                aoi_path, src, meta, src.crs)
        
        side_padding = (pred_window - step)//2
        mask = np.zeros((image[0].shape[0]+2*side_padding+pred_window,
                         image[0].shape[1]+2*side_padding+pred_window))

        for i in tqdm(range(mask.shape[-1]//step)):
            for j in range(mask.shape[-2]//step):

                width, heigth = int(i*step), int(j*step)
                img_chip = image[:, heigth:heigth+pred_window,
                                    width:width+pred_window]/255
                if img_chip.sum()<1:
                    continue
                    
                if (img_chip.shape[-1] != pred_window) or (
                    img_chip.shape[-2] != pred_window):
                    
                    img = np.zeros((image.shape[0], pred_window, pred_window))
                    img[:, :img_chip.shape[-2], :img_chip.shape[-1]] = img_chip
                else:
                    img = img_chip
                    
                pred = self._predict_chip(img, device, conf_thresh)
                pred = pred[
                    side_padding:pred_window-side_padding,
                    side_padding:pred_window-side_padding]*255
                    
                mask[heigth+side_padding:heigth+step+side_padding,
                     width+side_padding:width+step+side_padding] = pred
        mask = mask[:image[0].shape[0], :image[0].shape[1]]
        mask[aoi_bounds == 255] = 255
        
        meta['count'] = 1
        self._write_raster(mask, out_raster_path, meta)
        print(f'Writing raster: {out_raster_path}')
        
        return out_raster_path
    
    def process_raster_predictions(self, 
                                   raster_path, 
                                   shapes_path=None, 
                                   aoi_path=None,
                                   conf_thresh=0.5,
                                   dst_crs='EPSG:4326'):
        
        if shapes_path is None:
            shapes_path = raster_path.split('.')[0]+'_prediction.geojson'
        mask, crs, meta = self._load_tile(raster_path)
        if mask.shape[0]==1:
            mask = mask[0]

        _, contours = self.get_contours(mask, 75)
        polygons = self.polygonize(contours, meta)

        df = self._filter_polygons(polygons, aoi_path, src_crs=crs)
        df.to_crs(dst_crs, inplace=True)
        
        df.to_file(shapes_path, driver='GeoJSON')
        print(f'Writing resulting polygons: {shapes_path}')
        
        return df  
    
    def get_contours(self, mask, threshold=128, min_area=150):
        
        kernel_3 = np.ones((3,3), dtype=np.uint8)
        kernel_5 = np.ones((5,5), dtype=np.uint8)
        
        mask_dilated = cv2.dilate(mask, kernel_5)
        skeleton = morph.skeletonize(
            np.where(mask_dilated>threshold, 1, 0), method='lee')
        
        contours, _ = cv2.findContours(
            cv2.dilate((skeleton*255).astype(np.uint8), kernel_3),
            cv2.RETR_LIST, 
            cv2.CHAIN_APPROX_SIMPLE)
        
        #contours_skeleton = [c for c in contours_skeleton if c.area<min_area]
        filtered_conts = []
        
        for cont in contours:
            area = cv2.contourArea(cont)
            perimeter = cv2.arcLength(cont, True)
            fig = cv2.approxPolyDP(
                cont, (5e-4)*perimeter, True)
            
            if area>min_area:
                filtered_conts.append(fig)
        
        new_mask = np.zeros(mask.shape)
        new_mask = cv2.drawContours(new_mask, filtered_conts, -1, 255, 1)
        return new_mask, filtered_conts

    def polygonize(self, contours, meta, transform=True):
        """Credit for base setup: Michael Yushchuk. Thank you!"""
        polygons = []
        for i in tqdm(range(len(contours))):
            c = contours[i]
            n_s = (c.shape[0], c.shape[2])
            if n_s[0] > 2:
                if transform:
                    polys = [tuple(i) * meta['transform'] for i in c.reshape(n_s)]
                else:
                    polys = [tuple(i) for i in c.reshape(n_s)]
                polygons.append(Polygon(polys))
        return polygons

    @staticmethod
    def process_float(array):
        array = array.copy()
        array[array < 0] = 0
        array_ = np.uint8(array * 255)
        return array_

    @staticmethod
    def min_max(X, min, max):
        X_scaled = np.zeros(X.shape)
        for i in range(X.shape[0]):
            X_std = (X[i] - min[i]) / (max[i] - min[i])
            X_scaled[i] = X_std * (1 - 0) + 0

        return X_scaled


def save_polys_as_shp(polys, name):
    try:
        # Now convert it to a shapefile with OGR
        driver = ogr.GetDriverByName('Esri Shapefile')
        ds = driver.CreateDataSource(name)
        layer = ds.CreateLayer('', None, ogr.wkbPolygon)
        # Add one attribute
        layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
        defn = layer.GetLayerDefn()

        # If there are multiple geometries, put the "for" loop here
        for i in range(len(polys)):
            # Create a new feature (attribute and geometry)
            feat = ogr.Feature(defn)
            feat.SetField('id', i)

            # Make a geometry, from Shapely object
            geom = ogr.CreateGeometryFromWkb(polys[i].wkb)
            feat.SetGeometry(geom)

            layer.CreateFeature(feat)
            # feat = geom = None  # destroy these

        # Save and close everything
        # ds = layer = feat = geom = None
    except Exception:
        import geopandas as gpd
        frame = gpd.GeoDataFrame()
        frame.geometry = polys
        frame.to_file(name)