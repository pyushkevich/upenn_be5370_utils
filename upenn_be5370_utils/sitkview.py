"""
UPenn BE5370 Utilities

Functions to visualize 3D SimpleITK images
 
Written by Paul Yushkevich
Licensed under the MIT License
"""

import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


def special_colormap(name):
    if name == 'mask':
        return colors.ListedColormap([(0,0,0,0), 'red'])
    else:
        return name


def view_hemi(image, cursor=None, vmin=None, vmax=None, cmap='gray', alpha=0.5, name=None):
    """Display a 3D image in a layout similar to ITK-SNAP
    
    :param image: A 3D SimpleITK image. 
        Can also be a list/tuple of images that will be displayed as overlays.
    :param cursor: 
        3D coordinates of the crosshair in the image
    :param cmap: 
        Colormap (a string, see matplotlib documentation). Can also be a list/tuple. 
        In addition to standard colormaps, you can use 'mask' for plotting binary mask images.
    :param vmin: 
        Intensity value mapped to the beginning of the color map. Can also be a list/tuple.
    :param vmax: 
        Intensity value mapped to the end of the color map. Can also be a list/tuple.
    :param alpha: 
        Opacity for the overlays, only relevant when len(image) > 1
    :param name:  
        Name of the image and overlays to display on the colorbar, should be of the same size as image. 
    """

    # Is the image an array? If not make it an array of size 1
    if type(image) is not list and type(image) is not tuple:
        image = [ image ]

    # This helper function plots one slice from the image
    def plot_slice(ax, data, aspect, xhair, vmin, vmax, cmap='gray', title=None, alpha=1):
        # Is this RGB data? Then we have to transform it by vmin/vmax
        if len(data.shape) == 3 and data.shape[-1] == 3:
            im = ax.imshow((data - vmin) / (vmax - vmin), aspect=aspect, alpha=alpha)
        else:
            im = ax.imshow(data, cmap=cmap, aspect=aspect, vmin=vmin, vmax=vmax, alpha=alpha)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.axvline(x=xhair[0], color='lightblue')
        ax.axhline(y=xhair[1], color='lightblue')
        if title:
            ax.set_title(title, fontsize=8)
        return im

    # This helper function makes sure that options such as cmap, vmin, vmax
    # have the same size as image, repeating them if necessary
    def p_match_len(x):
        if type(x) is not list and type(x) is not tuple:
            return [x] * len(image)
        elif len(cmap) != len(image):
            return (x * len(image))[0:len(image)]
        else:
            return x

    # Is the color map an array of the same size as image array
    cmap = p_match_len(cmap)
    vmin = p_match_len(vmin)
    vmax = p_match_len(vmax)
    name = p_match_len(name)

    # Create the figure
    fig = plt.figure(layout="constrained", figsize=(8,4))

    # Figure out the relative width of the slices to plot
    w_sag = image[0].GetSpacing()[2] * image[0].GetSize()[2]
    w_cor = image[0].GetSpacing()[1] * image[0].GetSize()[1]

    # Width of the sagittal and coronal images in physical units
    axs = fig.subplots(1, 2, sharey='all', width_ratios=[w_sag, w_cor])
    axs[0].axis('off')
    axs[1].axis('off')

    # Color bar properties
    cb_spacing = 0.1 / (len(image) - 1) if len(image) > 1 else 0.
    cb_height = 0.9 / len(image) if len(image) > 1 else 1.

    # Plot image and overlays
    for j, im in enumerate(image):

        voxels = sitk.GetArrayFromImage(im)
        cursor = np.array(voxels.shape) // 2 if cursor is None else cursor
        sp = np.array(im.GetSpacing())

        param = { 
            'vmin': np.quantile(voxels, 0.001) if vmin[j] is None else vmin[j], 
            'vmax': np.quantile(voxels, 0.999) if vmax[j] is None else vmax[j], 
            'cmap': special_colormap(cmap[j]), 
            'alpha': 1.0 if j == 0 else alpha }

        im0 = plot_slice(axs[0], voxels.take(cursor[1], axis=1).swapaxes(0,1),
                         aspect=sp[2]/sp[0], xhair=(cursor[0],cursor[2]), title='Sagittal Plane', **param)
        im1 = plot_slice(axs[1], voxels.take(cursor[0], axis=0).swapaxes(0,1),
                         aspect=sp[2]/sp[1], xhair=(cursor[1],cursor[2]), title='Coronal Plane', **param)

        # Work out the position of the color bar
        cax = axs[1].inset_axes([1.04, j * (cb_spacing + cb_height), 0.05, cb_height])
        cbar = plt.colorbar(im1, orientation='vertical', ax=axs[1], cax=cax)
        auto_label = 'Main Image' if j == 0 else f'Overlay {j}'
        cbar.set_label(label=auto_label if name[j] is None else name[j], fontsize=8)
        cbar.ax.tick_params(labelsize=6)
        
    fig.show() 
