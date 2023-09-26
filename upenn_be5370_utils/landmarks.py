import SimpleITK as sitk
import numpy as np

def hemi_landmark_thickness(segmentation_image, landmark_image, gray_matter_label=1):
    """
    Compute the thickness of a cortical segmentation at specific landmarks.
    Inputs:
        segmentation_image: filename of a hemisphere segmentation
        landmark_image: filename of an image containing landmarks
        gray_matter_label: label of the gray matter in the segmentation (default: 1)
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
            grid_vox = np.stack(np.meshgrid(*[np.arange(k) for k in dot_roi.GetSize()]))
            A = np.array(dot_roi.GetDirection()).reshape(3,3) @ np.diag(dot_roi.GetSpacing())
            b = np.array(dot_roi.GetOrigin())
            grid_phy = np.einsum('ij,jxyz->ixyz', A, grid_vox) + b[:,None,None,None]
            dot_dist = np.sqrt(np.sum((grid_phy - np.array(dot_phy)[:,None,None,None])**2, 0))
            dot_dmap = sitk.Cast(sitk.GetImageFromArray(dot_dist), sitk.sitkFloat32)
            dot_dmap.CopyInformation(dot_roi)

            # Mask out the region on the skeleton where the distance from the dot is
            # less or equal to the radius, i.e., the dot is in the inscribed sphere
            dot_skel_mask = sitk.Cast(sitk.BinaryThreshold(seg_skel_thick - dot_dmap, 0, 1e100), sitk.sitkFloat32)

            # Report the thickness at this dot
            dot_thickness[labels[label]] = 2.0 * np.max(sitk.GetArrayFromImage(dot_skel_mask * seg_skel_thick))

    # Return the dots thicknesses
    return dot_thickness