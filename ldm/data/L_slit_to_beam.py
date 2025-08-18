import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.slit_data = None
        self.beam_data = None

    def __len__(self):
        return len(self.slit_data)

    def __getitem__(self, i):
        slit_example = self.slit_data[i]
        beam_example = self.beam_data[i]
        
        example = {
            'cond_image': slit_example['image'],
            'image': beam_example['image'],        
        }
        
        return example


class BeamOnlyTrain(Dataset):
    def __init__(self, size, beam_images_list_file, caption="slit-beam"):
        with open(beam_images_list_file, "r") as f:
            beam_paths = f.read().splitlines()
        assert len(beam_paths) > 0
        self.beam_data = ImagePaths(paths=beam_paths, size=size, random_crop=False)
        self.caption = caption  # <- 텍스트 프롬프트 고정

    def __len__(self):
        return len(self.beam_data)

    def __getitem__(self, i):
        beam_example = self.beam_data[i]  # {"image": Tensor[-1,1], "file_path_0": str, ...}
        return {
            "jpg": beam_example["image"],  # <- config.first_stage_key 와 일치
            "txt": self.caption            # <- config.cond_stage_key 와 일치
        }


class BeamOnlyVal(BeamOnlyTrain):
    pass

class BeamOnlyTest(BeamOnlyTrain):
    pass

        
        
        
        
        
        
        
                
