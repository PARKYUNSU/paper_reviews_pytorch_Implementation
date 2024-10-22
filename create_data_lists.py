from utils.utils import create_segmentation_data_lists

if __name__ == '__main__':
    create_segmentation_data_lists(voc07_path='/kaggle/input/pascal/archive/VOC2007',
                      voc12_path='/kaggle/input/pascal/archive/VOC2012',
                      output_folder='/kaggle/working/')
