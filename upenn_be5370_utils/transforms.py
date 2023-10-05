import SimpleITK as sitk
import numpy as np

# Load an ITK affine transform from the matrix format used by ITK-SNAP and greedy
def load_itksnap_transform(fn_affine):

    # Load the matrix and convert from RAS to LPS coordinate system
    affine = np.loadtxt(fn_affine)
    ras_to_lps = np.diag([-1,-1,1])
    A = ras_to_lps @ affine[:3,:3] @ ras_to_lps
    b = ras_to_lps @ affine[:3,3]

    # Create an ITK transform
    return sitk.AffineTransform(A.flatten().tolist(), b.tolist())


# Get the transform from voxel space to physical space for an image
def get_sitk_voxel_to_physical_transform(image):
    A = np.array(image.GetDirection()).reshape(3,3) @ np.diag(image.GetSpacing())
    b = np.array(image.GetOrigin())
    return A,b


# Convert an ITK affine transform from SimpleITK to PyTorch
def map_affine_sitk_to_pytorch(tran, img_fix, img_mov):
    """
    Compute an affine transform in PyTorch coordinate space that corresponds
    to a given SimpleITK affine transform.

    Args:
        tran: A SimpleITK Transform
        img_fix: The fixed image (SimpleITK Image)
        img_mov: The moving image (SimpleITK Image)
    Output:
        Q, p = map_affine_sitk_to_pytorch(tran, img_fix, img_mov)

        Q is the 3x3 affine matrix, p is the 3x1 translation vector that achieve
        the same result in PyTorch space as sitk.Resample(img_mov, img_fix, tran)
    """

    # Get the fixed and moving transforms
    M_f, o_f = get_sitk_voxel_to_physical_transform(img_fix)
    M_m, o_m = get_sitk_voxel_to_physical_transform(img_mov)

    # Get the affine parameters (assume center is not set)
    A = np.array(tran.GetMatrix()).reshape(3,3)
    b = np.array(tran.GetTranslation())

    # Get the mapping from index to PyTorch coordinate system
    sz_f, sz_m = np.array(img_fix.GetSize()), np.array(img_mov.GetSize())
    W_f, W_m = np.diag(2.0 / sz_f), np.diag(2.0 / sz_m)
    z_f, z_m = 1.0 / sz_f - 1.0, 1.0 / sz_m - 1.0

    # Compute the composed and inverse matrices
    W_f_inv = np.linalg.inv(W_f)
    M_m_inv = np.linalg.inv(M_m)

    # Compute the final transform
    Q = W_m @ M_m_inv @ A @ M_f @ W_f_inv
    p = W_m @ M_m_inv @ (A @ o_f + b - o_m) - Q @ z_f + z_m

    # Return the final transform
    return Q, p


# Get an affine transform that maps a coordinate in the sitk physical
# space to the correponding coordinate in the pytorch physical space
def get_physical_to_pytorch_coordinate_transform(img):
    """
    Compute the transform that maps points in the SimpleITK physical coordinate 
    system to the corresponding points in the PyTorch coordinate system.

    Args:
        img: An image for which the transform is computed
    Output:
        Q, p = get_physical_to_pytorch_coordinate_transform(img) returns a 
        3x3 matrix Q and a 3x1 vector p such that the PyTorch coordinates
        of a point with physical coordinates x_phys can be computed as

            x_torch = Q @ x_phys + p

    """
    # Get the fixed and moving transforms
    M, o = get_sitk_voxel_to_physical_transform(img)

    # The ITK/PyTorch coordinate swap
    K = np.fliplr(np.eye(3))

    # Get the mapping from index to PyTorch coordinate system
    sz = np.array(img.GetSize())
    W = np.diag(2.0 / sz)
    z = 1.0 / sz - 1.0

    # Compute the composed and inverse matrices
    M_inv = np.linalg.inv(M)

    # Compute the final transform
    Q = W @ K @ M_inv
    p = z - Q @ o

    # Return the final transform
    return Q, p


# Get an affine transform that maps a coordinate in the PyTorch coordinate
# system to a coordinate in ITK physical system
def get_pytorch_to_physical_coordinate_transform(img):
    """
    Compute the transform that maps points in the PyTorch coordinate system
    to the corresponding points in the SimpleITK physical coordinate system.

    Args:
        img: An image for which the transform is computed
    Output:
        Q, p = get_pytorch_to_physical_coordinate_transform(img) returns a 
        3x3 matrix Q and a 3x1 vector p such that the physical coordinates
        of a point with PyTorch coordinates x_torch can be computed as

            x_phys = Q @ x_torch + p

    """

    # Get the fixed and moving transforms
    M, s = get_physical_to_pytorch_coordinate_transform(img)
    Q = np.linalg.inv(M)
    p = -Q @ s
    return Q, p
