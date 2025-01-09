import tensorflow as tf
import keras
import os
import numpy as np
import cv2
import random

class DatasetVelho:
    CLASSES = ['metastasis']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=['metastasis'], 
            augmentation=None, 
            preprocessing=None,
            these_ids=None,
            start_index=None,
            end_index=None,
    ):
        if these_ids is None:
            self.ids = sorted(os.listdir(images_dir))
            if start_index is not None:
                self.ids = self.ids[start_index:end_index]
        else:
            self.ids = these_ids
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.see = False
    
    def __getitem__(self, i):
        #print(self.images_fps[i])
        image = cv2.imread(self.images_fps[i])
        #try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #except:
        #    print('-----------------------')
        #    print(i)
        #    print(self.images_fps[i])
        #    print('-----------------------')
        #    raise self.images_fps[i]
        mask = cv2.imread(self.masks_fps[i], 0)
        #mask = cv2.imread(self.masks_fps[0], 0)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        mask = (~mask.astype(bool)).astype(float)
        
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        size = (256, 256)
        image = cv2.resize(image, dsize=size, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, dsize=size, interpolation=cv2.INTER_CUBIC)
        mask = np.expand_dims(mask, axis=2)

        mask[mask >= 0.5] = 1.0
        mask[mask < 0.5] = 0.0
        
        return image/255.0, mask
        #return image/255, mask
        
    def __len__(self):
        return len(self.ids)
    
class DataloderVelho(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        
        #data = []
        image = []
        mask  = []
        
        for j in range(start, stop):
            #data.append(self.dataset[j])
            img, msk = self.dataset[j]
            image.append(img)
            mask.append(msk)
            
        image = np.array(image)
        mask  = np.array(mask)
        return image, mask
        
        # transpose list of lists
        #batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        #return batch
        #return batch[0], batch[1]
    
    def __len__(self):
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes) 

def split_canser_seg(types, split=0.5):
    all_ids = []
    for type_cancer in os.listdir('Cancer_Segmentation'):
        if type_cancer in types:
            ids = sorted(os.listdir('Cancer_Segmentation/'+type_cancer))
            ids = ['Cancer_Segmentation/'+type_cancer+'/'+one for one in ids if '_mask' not in one]
            all_ids = np.concatenate((all_ids, ids), axis=0)
    all_ids = np.random.permutation(all_ids)
    
    cut = int(split * len(all_ids))
    return all_ids[:cut], all_ids[cut:]

class DataloderCancerSeg(tf.keras.utils.Sequence):
    def __init__(self, types=[], ids=None, preprocess=None, batch_size=1, shuffle=False, encoding_constant=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.preprocess = preprocess
        self.stylizing = False
        self.encoding_constant = encoding_constant
        self.quick_eval = False

        if ids is None:
            all_ids = []
            for type_cancer in os.listdir('Cancer_Segmentation'):
                if type_cancer in types:
                    ids = sorted(os.listdir('Cancer_Segmentation/'+type_cancer))
                    ids = ['Cancer_Segmentation/'+type_cancer+'/'+one for one in ids if '_mask' not in one]
                    all_ids = np.concatenate((all_ids, ids), axis=0)

            self.indexes = all_ids
        else:
            self.indexes = ids

        self.on_epoch_end()

    def __getitem__(self, i):
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        
        image = []
        mask  = []
        
        for j in range(start, stop):
            if self.preprocess:
                image.append(self.preprocess(np.load(self.indexes[j])))
            else:
                image.append(np.load(self.indexes[j]))

            mask.append(np.load(self.indexes[j][:-4]+'_mask.npy'))
            
        image = np.array(image)
        mask  = np.array(mask)
        #if (self.encoding_constant is not None) and not self.quick_eval:
        #    mask = mask * self.encoding_constant

        if self.stylizing:
            return image, image
        return image, mask
    
    def __len__(self):
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        pass
        #if self.shuffle:
        #    self.indexes = np.random.permutation(self.indexes)   

def split_lynsec(split=0.5):
    all_ids = sorted(os.listdir('lynsec 1/'))
    all_ids = np.random.permutation(all_ids)
    cut = int(split * len(all_ids))
    return all_ids[:cut], all_ids[cut:]

class DataloderLynSec(tf.keras.utils.Sequence):
    def __init__(self, ids=None, preprocess=None, batch_size=1, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.preprocess = preprocess
        self.stylizing = False
        
        all_ids = []
        for id in ids:
            all_ids.append('lynsec 1/'+id)
        self.indexes = all_ids

        self.on_epoch_end()

    def __getitem__(self, i):
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        
        image = []
        mask  = []
        
        for j in range(start, stop):
            if self.preprocess:
                img = self.preprocess(np.load(self.indexes[j]))
            else:
                img = np.load(self.indexes[j])

            msk = img[:,:,4:5].copy()
            msk[msk != 2] = 0.0
            msk[msk != 0] = 1.0
            img = img[:,:,:3]
            
            size = (256, 256)
            img = cv2.resize(img.astype('float64'), dsize=size, interpolation=cv2.INTER_CUBIC)
            msk = cv2.resize(msk.astype('float64'), dsize=size, interpolation=cv2.INTER_CUBIC)
            msk = np.expand_dims(msk, axis=2)

            img[img > 255.0] = 255.0
            msk[msk >= 0.5] = 1.0
            msk[msk < 0.5] = 0.0

            image.append(img/255.0)
            mask.append(msk)
            
        image = np.array(image)
        mask  = np.array(mask)
        if self.stylizing:
            return image, image
        return image, mask
    
    def __len__(self):
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        pass
        #if self.shuffle:
        #    self.indexes = np.random.permutation(self.indexes)   

def split_bcss(split=0.5):
    all_ids = sorted(os.listdir('BCSS/'))
    all_ids = ['BCSS/'+one for one in all_ids if '_mask' not in one]
    all_ids = np.random.permutation(all_ids)
    cut = int(split * len(all_ids))
    return all_ids[:cut], all_ids[cut:]

class DataloderBCSS(tf.keras.utils.Sequence):
    def __init__(self, ids=None, preprocess=None, batch_size=1, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.preprocess = preprocess
        self.indexes = ids

        self.on_epoch_end()

    def __getitem__(self, i):
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        
        image = []
        mask  = []
        
        for j in range(start, stop):
            if self.preprocess:
                img = self.preprocess(np.load(self.indexes[j]))
            else:
                img = np.load(self.indexes[j])

            msk = np.load(self.indexes[j][:-4]+'_mask.npy')

            image.append(img)
            mask.append(msk)
            
        image = np.array(image)
        mask  = np.array(mask)
        return image, mask
    
    def __len__(self):
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        pass
        #if self.shuffle:
        #    self.indexes = np.random.permutation(self.indexes)   

def split_lung(split=0.5):
    all_ids = sorted(os.listdir('LungFCP/'))
    all_ids = ['LungFCP/'+one for one in all_ids if '_mask' not in one]
    all_ids = np.random.permutation(all_ids)
    cut = int(split * len(all_ids))
    return all_ids[:cut], all_ids[cut:]

class DataloderLung(tf.keras.utils.Sequence):
    def __init__(self, ids=None, preprocess=None, batch_size=1, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.preprocess = preprocess
        self.indexes = ids

        self.on_epoch_end()

    def __getitem__(self, i):
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        
        image = []
        mask  = []
        
        for j in range(start, stop):
            if self.preprocess:
                img = self.preprocess(np.load(self.indexes[j]))
            else:
                img = np.load(self.indexes[j])

            msk = np.load(self.indexes[j][:-4]+'_mask.npy')

            image.append(img)
            mask.append(msk)
            
        image = np.array(image)
        mask  = np.array(mask)
        return image, mask
    
    def __len__(self):
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        pass
        #if self.shuffle:
        #    self.indexes = np.random.permutation(self.indexes)   

class JoinDataSets(tf.keras.utils.Sequence):
    def __init__(self, datasets=[], maxlen=None, fulljoin=False, batchsize=None):
        self.datasets = datasets
        self.indexds = []
        self.lends = 0
        self.maxlen = maxlen

        self.fulljoin = fulljoin
        self.batchsize = batchsize
        self.minibatchsize = 0

        if self.fulljoin:
            if maxlen:
                self.lends = maxlen
            else:
                minlen = None
                for ds in self.datasets:
                    if (minlen is None) or (len(ds) < minlen):
                        minlen = len(ds)
                self.lends = minlen

            self.minibatchsize = int(self.batchsize / len(self.datasets))
        else:
            for ds in self.datasets:
                self.indexds.append(0)
                self.lends += maxlen if maxlen is not None else len(ds)

    def __getitem__(self, i):
        if self.fulljoin:
            iindex = i % self.lends

            batch_x = None
            batch_y = None
            for ds in self.datasets:
                data = ds[iindex]

                # sample N (minibatch) random elements from a batch
                #idxs = random.sample(range(0, self.batchsize-1), self.minibatchsize)
                idxs = range(0, self.minibatchsize)
                
                if batch_x is None:
                    batch_x = data[0][idxs]
                    batch_y = data[1][idxs]
                else:
                    batch_x = np.concatenate((batch_x, data[0][idxs]))
                    batch_y = np.concatenate((batch_y, data[1][idxs]))

            # batch not fill
            if len(batch_x) < self.batchsize: # mini batch is odd (odd number of datasets probably)
                #filtered_range = list(range(0, self.batchsize-1))
                #filtered_range = [i for i in filtered_range if i not in idxs]
                #new_idxs = random.sample(filtered_range, self.minibatchsize)
                new_idxs = range(self.minibatchsize, self.minibatchsize+(self.batchsize - len(batch_x)))
                
                batch_x = np.concatenate((batch_x, data[0][new_idxs]))
                batch_y = np.concatenate((batch_y, data[1][new_idxs]))

            # shuffle X,Y in same order
            #p = np.random.permutation(len(batch_x))    
            #return (batch_x[p], batch_y[p])
            return batch_x, batch_y

        else:
            dsindex = i % len(self.datasets)
            rep = 0
            while rep < len(self.datasets):
                iindex = self.indexds[dsindex]
                if iindex < (self.maxlen if self.maxlen is not None else len(self.datasets[dsindex])):
                    break
                dsindex = (dsindex+1) % len(self.datasets)
                rep += 1

            self.indexds[dsindex] = (self.indexds[dsindex]+1) % (self.maxlen if self.maxlen is not None else len(self.datasets[dsindex]))
            return self.datasets[dsindex][iindex]
    
    def __len__(self):
        return self.lends
    
    def on_epoch_end(self):
        if not self.fulljoin:
            self.indexds = []
            self.lends = 0
            for ds in self.datasets:
                self.indexds.append(0)
                self.lends += self.maxlen if self.maxlen is not None else len(ds)