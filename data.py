import os
import math
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
from scipy import ndimage

from random import randrange
from pydicom import dcmread
from rt_utils import RTStructBuilder

# AUX functions
def select_files(in_dir, filtername=None):
  files = []
  img_array = []
  for file in os.listdir(in_dir):
    if (filtername is not None) and (filtername not in file):
      continue
    files.append(os.path.join(in_dir, file))
  return files

def read_dicoms(in_dir, filtername=None):
  img_positions = []
  img_array = []
  for dcm in os.listdir(in_dir):
    if (filtername is not None) and (filtername not in dcm):
      continue
    ds = dcmread(os.path.join(in_dir, dcm))
    if 'Rows' in ds:
      img_array.append(ds.pixel_array.astype('float'))
      img_positions.append(ds.ImagePositionPatient[2])
  indexes = np.argsort(np.asarray(img_positions))
  out = np.asarray(img_array)[indexes]
  if len(out.shape) > 3:
    out = out[0]
  return out.transpose(1,2,0) # the Z is coming first, but should be the last index

def read_struct(cts_dir, struct_file, struct_name):
  rtstruct = RTStructBuilder.create_from(dicom_series_path=cts_dir, rt_struct_path=struct_file)
  return rtstruct.get_roi_mask_by_name(struct_name).astype('float')

def calculatePad(size1, size2):
  pad = (0,0)
  if size1 < size2:
    diff = (size2 - size1)
    half = diff // 2 #int division
    pad = (half, diff-half)
  return pad

def padImage(image, paddings):
  pad1 = calculatePad(image.shape[0], paddings[0])
  pad2 = calculatePad(image.shape[1], paddings[1])
  pad3 = calculatePad(image.shape[2], paddings[2])
  return np.pad(image, [pad1, pad2, pad3], mode='constant')

def calculatePaddings(shapes):
  shapes0 = []
  shapes1 = []
  shapes2 = []
  for shape in shapes:
    if shape is None:
      continue;
    shapes0.append(shape[0])
    shapes1.append(shape[1])
    shapes2.append(shape[2])

  return [np.max(shapes0), np.max(shapes1), np.max(shapes2)]

def normalize_volume(img):
  scaler = preprocessing.MinMaxScaler()
  img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
  return img, scaler

def resize_volume(img):
  desired_depth = 32
  desired_width = 128
  desired_height = 128

  current_width = img.shape[0]
  current_height = img.shape[1]
  current_depth = img.shape[2]

  width = current_width / desired_width
  height = current_height / desired_height
  depth = current_depth / desired_depth
  
  depth_factor = 1 / depth
  width_factor = 1 / width
  height_factor = 1 / height
  
  img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
  return img

def rotate3D(matrix, deg_angle, axis):
  d = len(matrix)
  h = len(matrix[0])
  w = len(matrix[0][0])
  min_new_x = 0
  max_new_x = 0
  min_new_y = 0
  max_new_y = 0
  min_new_z = 0
  max_new_z = 0
  new_coords = []
  angle = math.radians(deg_angle)
  print('1')
  for z in range(d):
    for y in range(h):
      for x in range(w):
        new_x = None
        new_y = None
        new_z = None

        if axis == "x":
          new_x = int(round(x))
          new_y = int(round(y*math.cos(angle) - z*math.sin(angle)))
          new_z = int(round(y*math.sin(angle) + z*math.cos(angle)))
        elif axis == "y":
          new_x = int(round(z*math.sin(angle) + x*math.cos(angle)))
          new_y = int(round(y))
          new_z = int(round(z*math.cos(angle) - x*math.sin(angle)))
        elif axis == "z":
          new_x = int(round(x*math.cos(angle) - y*math.sin(angle)))
          new_y = int(round(x*math.sin(angle) + y*math.cos(angle)))
          new_z = int(round(z))

        val = matrix.item((z, y, x))
        new_coords.append((val, new_x, new_y, new_z))
        if new_x < min_new_x: min_new_x = new_x
        if new_x > max_new_x: max_new_x = new_x
        if new_y < min_new_y: min_new_y = new_y
        if new_y > max_new_y: max_new_y = new_y
        if new_z < min_new_z: min_new_z = new_z
        if new_z > max_new_z: max_new_z = new_z
  print('2')
  new_x_offset = abs(min_new_x)
  new_y_offset = abs(min_new_y)
  new_z_offset = abs(min_new_z)

  new_width = abs(min_new_x - max_new_x)
  new_height = abs(min_new_y - max_new_y)
  new_depth = abs(min_new_z - max_new_z)
  print('3')
  rotated = np.empty((new_depth + 1, new_height + 1, new_width + 1))
  rotated.fill(0)
  print('4')
  for coord in new_coords:
    val = coord[0]
    x = coord[1]
    y = coord[2]
    z = coord[3]

    if rotated[new_z_offset + z][new_y_offset + y][new_x_offset + x] == 0:
      rotated[new_z_offset + z][new_y_offset + y][new_x_offset + x] = val

  return rotated

# Objects
class InstancePreprocessor:
  def __init__(self, in_dir, cts_filter=None, structs_filter=None, struct_name=None, oars_filter=None, doses_filter=None, batch_size=10):
    self.cases = select_files(in_dir)
    print('Instance size: ', len(self.cases))
    
    self.index = 0
    self.cts_filter=cts_filter
    self.structs_filter=structs_filter
    self.struct_name=struct_name
    self.oars_filter=oars_filter
    self.doses_filter=doses_filter
    self.batch_size=batch_size
      
  def reset(self, index=0):
    self.index = index
      
  def get(self, quantity):
    arr = []
    for i in range(quantity):
      try:
        dir_ct = select_files(self.cases[self.index+i], self.cts_filter)
        ct = read_dicoms(dir_ct[0])
      except Exception as e:
        print('Error reading CT case '+str(self.index+i))
        print(e)
        continue
      
      struct = None
      if self.structs_filter is not None:
        dir_struct = select_files(self.cases[self.index+i], self.structs_filter)
        struct_file = select_files(dir_struct[0])
        struct = read_struct(dir_ct[0], struct_file[0], self.struct_name)
          
      oar = None
      if self.oars_filter is not None:
        dir_oar = select_files(self.cases[self.index+i], self.oars_filter)
        oar = read_dicoms(dir_oar[0])
      
      try:
        dir_dose = select_files(self.cases[self.index+i], self.doses_filter)
        dose = read_dicoms(dir_dose[0])
      except Exception as e:
        print('Error reading DOSE case '+str(self.index+i))
        print(e)
        continue
      
      arr.append([ct, struct, oar, dose])
        
    self.index = self.index + quantity
    return arr
  
  def calculate(self):
    print('Calculating...')
    shapes = []
    for _ in range(0, len(self.cases)):
      cases = self.get(1)
      if len(cases) == 0:
        continue
      for img in cases[0]:
        if img is None:
          continue;
        shapes.append(img.shape)

    self.shape_paddings = calculatePaddings(shapes)
    print('Instance dimension to pad: ', self.shape_paddings)
    self.reset()

  def preprocess(self, out_dir, data_augmentation=0):
    for i in range(0, len(self.cases), self.batch_size):
      print('Sampling '+str(self.batch_size))
      samples = self.get(self.batch_size)
      print('Processing...')
      for sample in samples:
        sample[0] = padImage(sample[0], self.shape_paddings)
        sample[0], scalerCT = normalize_volume(sample[0])
        
        if self.structs_filter is not None:
          sample[1] = padImage(sample[1], self.shape_paddings)
        else:
          sample[1] = np.asarray([])
            
        if self.oars_filter is not None:
          sample[2] = padImage(sample[2], self.shape_paddings)
        else:
          sample[2] = np.asarray([])

        sample[3] = padImage(sample[3], self.shape_paddings)
        sample[3], scalerDose = normalize_volume(sample[3])

        filename = os.path.join(out_dir, 'case'+str(i))
        np.savez_compressed(filename, CT=sample[0], PTV=sample[1], OAR=sample[2], DOSE=sample[3])

        with open(filename+'.scalers', 'wb') as handle:
          pickle.dump({'CT':scalerCT, 'DOSE':scalerDose}, handle)
        
        if data_augmentation > 0:
          print('Augmenting by '+str(data_augmentation))
          for j in range(data_augmentation):
            randomx = randrange(300)
            randomy = randrange(300)
            randomz = randrange(300)
            print(randomx, randomy, randomz)
            
            aa = rotate3D(sample[0], randomx, "x")
            print(aa.shape)
            raise 'aaa'
            sample0 = rotate3D(rotate3D(rotate3D(sample[0], randomx, "x"), randomy, "y"), randomz, "z")
            if (sample0==sample[0]).all():
              print('0 igual')
            if self.structs_filter is not None:
              sample1 = rotate3D(rotate3D(rotate3D(sample[1], randomx, "x"), randomy, "y"), randomz, "z")
              if (sample1==sample[1]).all():
                print('1 igual')
            if self.oars_filter is not None:
              sample2 = rotate3D(rotate3D(rotate3D(sample[2], randomx, "x"), randomy, "y"), randomz, "z")
              if (sample2==sample[2]).all():
                print('2 igual')
            sample3 = rotate3D(rotate3D(rotate3D(sample[3], randomx, "x"), randomy, "y"), randomz, "z")
            if (sample3==sample[3]).all():
              print('3 igual')
            
            filename = os.path.join(out_dir, 'case'+str(i)+'_'+str(j))
            np.savez_compressed(filename, CT=sample0, PTV=sample1, OAR=sample2, DOSE=sample3)
          raise 'aaa'

class InstanceGenerator(tf.keras.utils.Sequence):
  def __init__(self, cases, batch_size):
    self.cases = cases
    self.batch_size = batch_size
    self.X = []
    self.Y = []
    self.chamadas = []

  def batch(self, batch_size):
    self.batch_size = batch_size

  def preload(self):
    print('Preloading cases ', str(len(self.cases)))
    for i in range(len(self.cases)):
      try:
        case = np.load(self.cases[i])
        X = np.array([resize_volume(case['CT']), resize_volume(case['PTV'])])
        Y = np.array([resize_volume(case['DOSE'])])
        self.X.append(X.transpose(1,2,3,0))
        self.Y.append(Y.transpose(1,2,3,0))
      except Exception as e:
        print('Error reading case '+str(i))
        print(e)
        continue

  def __len__(self):
    return int(len(self.X)/self.batch_size)

  def __getitem__(self, idx):
    low = idx * self.batch_size
    high = min(low + self.batch_size, len(self.cases))

    arrX = []
    arrY = []
    for i in range(low, high):
      arrX.append(self.X[i])
      arrY.append(self.Y[i])
      #try:
      #  case = np.load(self.train_cases[i] if self.trainMode else self.test_cases[i])
      #  arrX.append(resize_volume(case['CT']))
      #  arrY.append(resize_volume(case['DOSE']))
      #except Exception as e:
      #  print('Error reading case '+str(+i))
      #  print(e)
      #  continue
    self.chamadas.append((idx, len(arrX)))
    return np.asarray(arrX), np.asarray(arrY)

class InstanceLoader():
  def __init__(self, in_dir, batch_size=5, split_test=0.2):
    self.batch_size = batch_size
    self.cases = select_files(in_dir,'.npz')

    train_cases, test_cases = train_test_split(self.cases, test_size=split_test, random_state=0)
    self.train_cases = train_cases
    self.test_cases = test_cases

    self.trainGen = InstanceGenerator(self.train_cases, self.batch_size)
    self.testGen = InstanceGenerator(self.test_cases, self.batch_size)

    print('Instance size:', len(self.cases))
    print('Train size:', len(self.train_cases))
    print('Test size:', len(self.test_cases))

  def shape(self):
    case = np.load(self.cases[0])
    return resize_volume(case['CT']).shape
    
  def getCase(self, index=0):
    return np.load(self.cases[index])

  def batch(self, batch_size):
    self.trainGen.batch(batch_size)
    self.testGen.batch(batch_size)

  def preload(self):
    self.trainGen.preload()
    self.testGen.preload()