from albumentations import *
import random
import numpy as np


MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}

class ChannelShift(ImageOnlyTransform):
    """Randomly shift values for each channel of the input image.
    Args:
        shift: scalar - uniform distribution from (-shift,shift)
                   across all channels.
               tuple(scalar, scalar) - uniform shift drawn from the tuple
                   across all channels.
               list or array-like of scalars - for channel i the shift would be
                   drawn from (-shift[i],shift[i]). The length of the argument must
                   match the number of channels.
               array-like of tuples(scalar,scalar) - defines shift policy for
                   each channel. The length of the argument must
                   match the number of channels.
        channels (int): the number of channels in the image. Only used while
            broadcasting shift generation.
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        shift=20,
        channels=3,
        always_apply=False,
        p=0.5,
    ):
        super(ChannelShift, self).__init__(always_apply, p)
        self.shift = shift
        self.channels = channels
    
    def apply(self, image, shift=0, **params):
        if image.shape[-1]!=len(shift):
            raise ValueError ('Shift parameter does not match the number of channels.')
        out_dtype = image.dtype
        image = image.astype(np.float32)
        out = np.empty_like(image)
        
        for i in range(len(shift)):
            out[...,i] = image[...,i] + shift[i]
        
        out = np.clip(out,0,MAX_VALUES_BY_DTYPE.get(out_dtype,1.0)).astype(out_dtype)
        return out

    def get_params(self):
        if isinstance(self.shift,tuple):
            return {'shift': [random.uniform(*self.shift)
                              for i in range(self.channels)]}
        arg_type = len(np.array(self.shift).shape)
        if arg_type==0:
            return {'shift': [random.uniform(-self.shift,self.shift)
                              for i in range(self.channels)]}
        
        if arg_type==1:
            return {'shift': [random.uniform(-s,s) for s in self.shift]}
        
        if arg_type==2:
            return {'shift': [random.uniform(-l,h) for l,h in self.shift]}

        
        raise ValueError('Shift parameter could not be interpreted.\n Please, check the transformation docstring.')

    def get_transform_init_args_names(self):
        return ("shift",)