import os
import codecs
import numpy as np
import pickle
from skimage.io import imsave

data_path = 'MNIST/'
files = os.listdir(data_path)


def get_int(binary):
    return int(codecs.encode(binary, 'hex'), 16)


data_dict = {}
for file in files:
    if file.endswith('ubyte'):
        print('Reading ', file)
        with open(data_path+file, 'rb') as f:
            data = f.read()
            file_type = get_int(data[:4])
            length = get_int(data[4:8])
            if file_type == 2051:
                category = 'images'
                num_rows = get_int(data[8:12])
                num_cols = get_int(data[12:16])
                parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
                parsed = parsed.reshape(length, num_rows, num_cols)
            elif file_type == 2049:
                category = 'labels'
                parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
                parsed = parsed.reshape(length)
            if length == 10000:
                data_set = 'test'
            elif length == 60000:
                data_set = 'train'
            data_dict[data_set+'_'+category] = parsed

# print(data_dict)
pickle.dump(data_dict['train_images'], open('train_images.pkl', 'wb'))
pickle.dump(data_dict['train_labels'], open('train_labels.pkl', 'wb'))
pickle.dump(data_dict['test_images'], open('test_images.pkl', 'wb'))
pickle.dump(data_dict['test_labels'], open('test_labels.pkl', 'wb'))


# data_sets = ['train', 'test']
#
# for each_set in data_sets:
#     images = data_dict[each_set+'_images']
#     labels = data_dict[each_set+'_labels']
#     no_of_samples = images.shape[0]
#     for index in range(no_of_samples):
#         print(each_set, index)
#         image = images[index]
#         label = labels[index]
#         if not os.path.exists(data_path+each_set+'/'+str(label)+'/'):
#             os.makedirs(data_path+each_set+'/'+str(label)+'/')
#         file_number = len(os.listdir(data_path+each_set+'/'+str(label)+'/'))
#         imsave(data_path+each_set+'/'+str(label)+'/%05d.png' % file_number, image)


