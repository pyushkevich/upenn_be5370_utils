import setuptools

setuptools.setup(
    name = 'upenn_be5370_utils',
    packages = setuptools.find_packages(),
    version = '0.2',
    license="MIT",
    description = 'Python routines for working with 3D images in UPenn BE5370 homework assignments',
    author = 'Paul A. Yushkevich <pyushkevich@gmail.com>',
    url = 'https://github.com/pyushkevich/upenn_be5370_utils',
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Visualization',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'SimpleITK', 
        'matplotlib', 
        'numpy'
    ]
)