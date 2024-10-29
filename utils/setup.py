from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='densecrf_gpu',
    ext_modules=[
        CUDAExtension('densecrf_gpu', [
            'densecrf_gpu.cu',  # CUDA 소스 파일
            'pairwise_gpu.cu'   # 관련 CUDA 소스 파일
        ])
    ],
    cmdclass={'build_ext': BuildExtension}
)