import SimpleITK as sitk
import numpy as np

def make_phys_coord_grid(image):
    grid_vox = np.stack(np.meshgrid(*[np.arange(k) for k in image.GetSize()]))
    A = np.array(image.GetDirection()).reshape(3,3) @ np.diag(image.GetSpacing())
    b = np.array(image.GetOrigin())
    grid_phy = np.einsum('ij,jxyz->ixyz', A, grid_vox) + b[:,None,None,None]
    grid_itk = sitk.GetImageFromArray(grid_phy.transpose(3,1,2,0), True)
    grid_itk.CopyInformation(image)
    return sitk.Cast(grid_itk, sitk.sitkVectorFloat32)

def point_distance_image(grid_phy, point):
    grid_x = sitk.VectorIndexSelectionCast(grid_phy, 0)
    grid_y = sitk.VectorIndexSelectionCast(grid_phy, 1)
    grid_z = sitk.VectorIndexSelectionCast(grid_phy, 2)
    return sitk.Sqrt((grid_x - point[0])**2 + (grid_y - point[1])**2 + (grid_z - point[2])**2)

def hemi_landmark_thickness(
    segmentation_image, landmark_image, 
    gray_matter_label = 1, sphere_image = None,
    intermediate_outputs_base_path = None):
    """
    Compute the thickness of a cortical segmentation at specific landmarks.
    Inputs:
        segmentation_image: filename of a hemisphere segmentation
        landmark_image: filename of an image containing landmarks
        gray_matter_label: label of the gray matter in the segmentation (default: 1)
        sphere_image: optional output image to save maximum inscribed spheres
        intermediate_outputs_base_path: 
            if provided, intermediate outputs from the thickness computation will be
            saved using the provided path as the prefix. For debugging.
    Returns:
        dict containing thickness values at specific landmarks
    """

    # Read the two images
    seg=sitk.ReadImage(segmentation_image, sitk.sitkUInt8)
    dots=sitk.ReadImage(landmark_image, sitk.sitkUInt8)

    # The labels that we care about
    labels = {
        2: 'MotorCortex',
        4: 'MedialFrontalCortex',
        10: 'AnteriorTemporalPole'
    }

    # Check that the image dimensions are the same
    if seg.GetSize() != dots.GetSize():
        raise ValueError("The segmentation and dots images must have the same size!")
    
    # Extract just the gray matter segmentation
    seg_bin = sitk.BinaryThreshold(seg, gray_matter_label, gray_matter_label)
            
    # If sphere image requested, compute a mesh grid for it
    if sphere_image:
        sphere_grid = make_phys_coord_grid(dots)
        spheres = dots * 0. 

    # Find the center of each dot in the image
    stat_filter = sitk.LabelShapeStatisticsImageFilter()
    stat_filter.Execute(dots)
    dot_thickness = {}
    for label in labels.keys():
        if label in stat_filter.GetLabels():
            # Find dot center in image coordinates
            dot_phy = stat_filter.GetCentroid(label)
            dot_vox = dots.TransformPhysicalPointToIndex(dot_phy)

            # Generate a region for computation
            reg_size = [65, 65, 65]
            reg_index = [ dot_vox[d] - 32 for d in range(3) ]

            # Make sure the region fits inside of the image region
            for d in range(3):
                if reg_index[d] < 0:
                    reg_size[d] += reg_index[d]
                    reg_index[d] = 0
                if reg_index[d] + reg_size[d] > dots.GetSize()[d]:
                    reg_size[d] += dots.GetSize()[d] - (reg_index[d] + reg_size[d])

            dot_roi = sitk.RegionOfInterest(dots, reg_size, reg_index)
            seg_roi = sitk.RegionOfInterest(seg, reg_size, reg_index)

            # Compute the skeleton of the segmentation
            seg_skel = sitk.BinaryThinning(seg_roi)

            # Compute the distance map from the boundary, in physical units
            seg_dmap = sitk.SignedDanielssonDistanceMap(seg_roi, True, False, True)

            # Multiply the skeleton by the thickness - this gives us the thickness
            # at each skeleton point
            seg_skel_thick = sitk.Cast(seg_skel, sitk.sitkFloat32) * seg_dmap

            # Compute the distance from the dot in physical units
            grid_phy = make_phys_coord_grid(dot_roi) 
            dot_dmap = point_distance_image(grid_phy, dot_phy)

            # Mask out the region on the skeleton where the distance from the dot is
            # less or equal to the radius, i.e., the dot is in the inscribed sphere
            dot_skel_mask = sitk.Cast(sitk.BinaryThreshold(seg_skel_thick - dot_dmap, 0, 1e100), sitk.sitkFloat32)

            # Report the thickness at this dot
            masked_skel = sitk.GetArrayFromImage(dot_skel_mask * seg_skel_thick)
            dt = 2.0 * np.max(masked_skel)
            dot_thickness[labels[label]] = dt

            # Save intermediate outputs
            if intermediate_outputs_base_path:
                sitk.WriteImage(seg_skel_thick, f'{intermediate_outputs_base_path}_{label:03d}_skel.nii.gz')
                sitk.WriteImage(dot_dmap, f'{intermediate_outputs_base_path}_{label:03d}_dot_dmap.nii.gz')
                sitk.WriteImage(dot_skel_mask, f'{intermediate_outputs_base_path}_{label:03d}_dot_skel_mask.nii.gz')

            # Fill out the sphere image if requested
            if sphere_image:
                mib_center_vox = list(reversed([ int(x) for x in np.unravel_index(np.argmax(masked_skel), masked_skel.shape) ]))
                print(mib_center_vox)
                mib_center_phy = dot_roi.TransformIndexToPhysicalPoint(mib_center_vox)
                mib_dmap = point_distance_image(sphere_grid, mib_center_phy)
                img_sphere = sitk.BinaryThreshold(mib_dmap - dt / 2, -1e10, 0, label, 0)
                spheres = sitk.Add(spheres, img_sphere)
                print('added')

    if sphere_image:
        spheres = sitk.Cast(spheres, sitk.sitkUInt8)
        sitk.WriteImage(spheres, sphere_image)

    # Return the dots thicknesses
    return dot_thickness