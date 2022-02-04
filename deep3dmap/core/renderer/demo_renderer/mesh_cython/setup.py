'''
python setup.py build_ext -i
to compile
'''

# setup.py
from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
# # setup(ext_modules = cythonize(Extension(
# #     'render_texture',
# #     sources=['render_texture.pyx'],
# #     language='c++',
# #     include_dirs=[numpy.get_include()],
# #     library_dirs=[],
# #     libraries=[],
# #     extra_compile_args=[],
# #     extra_link_args=[]
# # )))

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("render_cython",
                 sources=["render_cython.pyx", "render.cpp"],
                 language='c++',
                 include_dirs=[numpy.get_include()])],
)

# from distutils.core import setup
# from Cython.Build import cythonize

# setup(ext_modules = cythonize(
#            "_render.pyx",                 # our Cython source
#            sources=["render.c"],  # additional source file(s)
#            language="c",             # generate C++ code
#            include_dirs=[numpy.get_include()]
#       ))