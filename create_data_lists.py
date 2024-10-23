from utils.utils import create_segmentation_data_lists

if __name__ == '__main__':
    create_segmentation_data_lists(voc07_path='/kaggle/input/voc0712/VOC_dataset/VOCdevkit/VOC2007',
                      voc12_path='/kaggle/input/voc0712/VOC_dataset/VOCdevkit/VOC2012',
                      output_folder='/kaggle/working/')
