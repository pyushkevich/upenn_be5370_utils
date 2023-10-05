"""
UPenn BE5370 Utilities

Functions to visualize 3D SimpleITK images
 
Written by Paul Yushkevich
Licensed under the MIT License
"""

import SimpleITK as sitk
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec


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


def sitk_check_same_space(img, ref):
    
    # Check the size
    if img.GetSize() != ref.GetSize():
        return False
    
    # Extract the first pixel
    def get_dummy(x):
        d = x[0:1,0:1,0:1]
        d = sitk.VectorIndexSelectionCast(d) if d.GetNumberOfComponentsPerPixel() > 1 else d
        return sitk.Cast(d, sitk.sitkFloat32)
    
    try: 
        get_dummy(ref) + get_dummy(img)
        return True
    except Exception:
        return False
    

def view_sitk(*images, cursor=None, vmin=None, vmax=None, cmap='gray', alpha=0.5, 
              name=None, width=None, layout='overlay', title=None):
    """
    Display a 3D SimpleITK image in a layout similar to ITK-SNAP.

    Args:
        *images: 
            One or more 3D SimpleITK images to display. Images can be single-component
            or 3-component images. If 3-component, images are treated as RGB images.
        cursor: 
            The 3D coordinates of the crosshair in the image, defaults to the center 
            of the image.
        cmap:
            Color map, a string or a list/tuple of strings of the same size as `image. 
            In addition to the standard `matplotlib` colormaps, you can use `mask`
            for plotting binary images as overlays. Also, if the colormap is a qualitative
            color map, like `tab20`, the zero intensity will be made transparent and the
            corresponding image will be plotted without interpolation.
        vmin, vmax:
            Intensity value mapped to the beginning/end of the color map. A number or 
            a list/tuple of numbers of the same size as `image`.
        layout: 
            How to lay out multiple images (i.e., when image is a list/tuple). Use 'o' or 'overlay' 
            to plot images on top of each other with transparency, and 't' or 'tile' to tile images
            next to each other. You can also specify an array of strings matching the number of images,
            so that some images are tiled and some images are overlaid. You can also pass in a string
            like 'ttt' or 'tot' instead of an array of strings.
        alpha:
            Opacity for the 'overlay' mode
        width:
            The width of the figure, in inches. Same as calling set_figwidth() on the 
            figure returned by this function.
        name:
            Name of the image and overlays to display on the colorbar, a string or 
            a list/tuple of strings. 
        title:
            Overall title for the plot.

    Returns:
        fig, ax = view_sitk(image, ...) returns the handle to the figure and axes in 
        the same way as matplotlib.pyplot.subfigure
    """

    # Get the number of images
    nimg = len(images)

    # Check that the images are all SimpleITK images
    if nimg == 0 or not all((isinstance(x, sitk.Image) and x.GetDimension()==3 for x in images)):
        raise ValueError('Inputs to view_sitk must be 3D SimpleITK images')

    def plot_anat(ax, img, voxels, plane, is_discrete, cursor, **kwargs):
        if plane == 'axial':
            ax_rcs, dir_rcs = [1, 0, 2], [-1, 1, 1]
        elif plane == 'coronal':
            ax_rcs, dir_rcs = [2, 0, 1], [1, 1, 1]
        elif plane == 'sagittal':
            ax_rcs, dir_rcs = [2, 1, 0], [1, 1, 1]
        M = np.array(img.GetDirection()).reshape(3,3) @ np.diag(img.GetSpacing())
        Minv = np.linalg.inv(M)
        axmap = np.argmax(np.abs(Minv), 0)
        axsign = np.sign(Minv)[axmap].diagonal()
        a_r, a_c, a_s = axmap[ax_rcs]
        s_r, s_c, s_s = axsign[ax_rcs]

        # Extract the correct slice
        if img.GetNumberOfComponentsPerPixel() == 1:
            sl = voxels.transpose(2,1,0).transpose(a_r, a_c, a_s)[:,:,cursor[a_s]]
        else:
            sl = voxels.transpose(2,1,0,3).transpose(a_r, a_c, a_s, 3)[:,:,cursor[a_s],:]

        # Flip the axes correctly based on the direction
        sl = np.flipud(sl) if s_r * dir_rcs[0] < 0 else sl
        sl = np.fliplr(sl) if s_c * dir_rcs[1] < 0 else sl

        # Compute the transform from slice index to imshow coordinates
        spc = np.array(img.GetSpacing())[[a_r, a_c]]
        dim = np.array(img.GetSize())[[a_r, a_c]]
        ext = dim * spc
        if is_discrete:
            q = ax.imshow(np.ma.masked_equal(sl, 0), origin='lower', 
                        extent=(-spc[1]/2, ext[1] - spc[1]/2, -spc[0]/2, ext[0] - spc[0]/2), 
                        aspect=1, interpolation='none', **kwargs)
        else:
            q = ax.imshow(sl, origin='lower', 
                        extent=(-spc[1]/2, ext[1] - spc[1]/2, -spc[0]/2, ext[0] - spc[0]/2), 
                        aspect=1, **kwargs)
        ax.axvline(x=cursor[a_c] * spc[1], color='lightblue')
        ax.axhline(y=cursor[a_r] * spc[0], color='lightblue')
        return q, ext

    # This helper function makes sure that options such as cmap, vmin, vmax
    # have the same size as image, repeating them if necessary
    def p_match_len(x):
        if type(x) is not list and type(x) is not tuple:
            return [x] * nimg
        elif len(x) != nimg:
            return (x * nimg)[0:nimg]
        else:
            return x

    # Is the color map an array of the same size as image array
    cmap = p_match_len(cmap)
    vmin = p_match_len(vmin)
    vmax = p_match_len(vmax)
    name = p_match_len(name)

    # Process the layout
    if type(layout) is str and layout not in ['tile', 'overlay', 't', 'o']:
        layout = [ x for x in layout]
    layout = p_match_len(layout)

    # Create the figure
    fig = plt.figure(layout="constrained", figsize=(8,4))

    # Form the layout. First find all the images that should be shown as separate tiles
    tiles = [ [j] for j,l in enumerate(layout) if j==0 or l in ('t', 'tile')]

    # Now for each tile add overlays
    overlays = [ j for j,l in enumerate(layout) if j>0 and l in ('o', 'overlay') ]
    tiles = [ x + overlays for x in tiles ]

    # Create a grid spec, which we will adjust later based on the size of the images
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[0.8, 0.2], hspace=0.03)

    # Within each grid spec, create grid specs for the tiles
    gss = [gridspec.GridSpecFromSubplotSpec(1, len(tiles), subplot_spec=gs[0,j]) for j in range(2) ]

    # Create axes for each of the subplots
    axs = [ [] for j in (0,1) ]
    for j in range(2):
        for k in range(len(tiles)):
            ax_jk = fig.add_subplot(gss[j][k]) if j+k == 0 else fig.add_subplot(gss[j][k], sharey=axs[0][0])
            ax_jk.axis('off')
            axs[j].append(ax_jk)
            
    # Color bar properties
    cb_spacing = 0.1 / (nimg - 1) if nimg > 1 else 0.
    cb_height = 0.9 / nimg if nimg > 1 else 1.

    # Resample the images into the space of the main image if they don't occupy the same space
    main = images[0]
    iresam = [ sitk.Resample(im, main) if j > 0 and not sitk_check_same_space(im, main) 
                                       else im for j, im in enumerate(images) ]
    
    # Extract voxel arrays for all images 
    voxels = [ sitk.GetArrayFromImage(im) for im in iresam ]

    # Compute vmin/vmax that will be used for each of the images
    vmin = [ np.quantile(v, 0.001) if vmin[j] is None else vmin[j] for j,v in enumerate(voxels) ]
    vmax = [ np.quantile(v, 0.999) if vmax[j] is None else vmax[j] for j,v in enumerate(voxels) ]
    
    # Preserve the original range for colorbar display
    vmin_adj, vmax_adj = [], []

    # Adjust vmin/vmax for RGB images, because imshow only allows them to be in 0-1 range
    for j in range(len(voxels)):
        if iresam[j].GetNumberOfComponentsPerPixel() == 3:
            voxels[j] = np.clip((voxels[j] - vmin[j]) / (vmax[j] - vmin[j]), 0, 1)
            vmin_adj.append(0)
            vmax_adj.append(1)
        else:
            vmin_adj.append(vmin[j])
            vmax_adj.append(vmax[j])

    # Get the first (main) image and cursor position
    cursor = np.array(main.GetSize()) // 2 if cursor is None else cursor

    # Iterate over the tiles
    for t in range(len(tiles)):

        # Iterate over the images in that tile
        for i, j in enumerate(tiles[t]):
            param = { 
                'vmin': vmin_adj[j], 'vmax': vmax_adj[j], 
                'cmap': special_colormap(cmap[j]), 
                'alpha': 1.0 if i == 0 else alpha }

            # Get the type of the color map and assign it to the is_discrete array
            is_discrete = isinstance(plt.get_cmap(param['cmap']), matplotlib.colors.ListedColormap)

            # Get the extents of the plot
            q_s, ext_s = plot_anat(axs[0][t], iresam[j], voxels[j], 'sagittal', is_discrete, cursor, **param)
            q_c, ext_c = plot_anat(axs[1][t], iresam[j], voxels[j], 'coronal', is_discrete, cursor, **param)

            # Adjust the grid widths and figure widths for the top level grid
            if t == 0 and i == 0:
                gs.set_width_ratios([ext_s[1], ext_c[1]])
        
    # Generate the color bars
    gss_cb = gridspec.GridSpecFromSubplotSpec(1, len(iresam), subplot_spec=gs[1,:])
    for j in range(len(iresam)):
        ax_cb = fig.add_subplot(gss_cb[j])
        ax_cb.get_yaxis().set_visible(False)
        ax_cb.imshow(np.linspace(vmin_adj[j], vmax_adj[j], 256)[None,:], 
                     cmap=cmap[j], vmin=vmin_adj[j], vmax=vmax_adj[j], aspect='auto',
                     extent=(vmin[j], vmax[j], 0, 1))
        auto_label = 'Main Image' if j == 0 else f'Overlay {j}'
        ax_cb.set_title(auto_label if name[j] is None else name[j], fontsize=10)
        
    # Add markers to the coronal and sagittal views
    def marker(ax, text, pos):
        param = {'left': (0.01, 0.5, 'left', 'center'),
                 'right': (0.99, 0.5, 'right', 'center'),
                 'top': (0.5, 0.99, 'center', 'top'),
                 'bottom': (0.5, 0.01, 'center', 'bottom')}[pos]
        ax.text(param[0], param[1], text, 
                horizontalalignment=param[2], verticalalignment=param[3], 
                color='lightgreen', size=14, 
                transform=ax.transAxes,
                path_effects=[pe.withStroke(linewidth=3, foreground="black")])
        
    marker_pos = [
        { 'A': 'left', 'S': 'top', 'P': 'right', 'I': 'bottom' },
        { 'R': 'left', 'S': 'top', 'L': 'right', 'I': 'bottom' }
    ]
    
    for t in range(len(tiles)):
        for i, mp in enumerate(marker_pos):
            for k, v in mp.items():
                marker(axs[i][t], k, v)

    # Add title at the top
    if title:
        fig.suptitle(title)

    # Adjust width and height of the figure, leaving a fixed space for the colorbar
    if width:
        fig.set_figwidth(width) 
    else:
        width = fig.get_figwidth()
    height_tiles = ext_s[0] * width / (len(tiles) * (ext_s[1] + ext_c[1]))
    height_colorbars = 0.7
    fig.set_figwidth(width) 
    fig.set_figheight(height_tiles + height_colorbars + (0.3 if title else 0.0))
    gs.set_height_ratios([height_tiles, height_colorbars])

    return fig, axs