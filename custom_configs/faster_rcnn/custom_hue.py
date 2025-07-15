import random
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
from mmcv.image import adjust_hue


@TRANSFORMS.register_module()
class CustomHue(BaseTransform):
    """Rotates the hue channel within 45 degrees ensuring a divcersity in the blue colour

    Args:
        p (float): Probability of shifts.


    """

    def __init__(self, prob):
        self.prob = prob
   
    
    def transform(self, results):
        if random.random() < self.prob:
            img = results['img']
            value = random.uniform(0, 0.126)
            results['img'] = adjust_hue(img, value).astype(img.dtype)
        
        return results


