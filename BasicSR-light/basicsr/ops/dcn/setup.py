import os
os.environ['CUDA_HOME'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='deform_conv_ext',
    ext_modules=[
        CUDAExtension(
            name='basicsr.ops.dcn.deform_conv_ext',
            sources=[
                'src/deform_conv_ext.cpp',
                'src/deform_conv_cuda.cpp',
                'src/deform_conv_cuda_kernel.cu'
            ],
            extra_compile_args={
                'cxx': ['/EHsc'],
                'nvcc': ['-O2']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
