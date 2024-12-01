import os
import numpy as np
import pydicom
import nibabel as nib
from skimage.draw import polygon
from collections import Counter


def convert_dicom_to_nifti(dicom_dir, output_path):
    """
    Convert a DICOM series to a NIfTI file with nibabel, 
    preserving aspect ratio.
    """
    dicom_files = sorted(
        [
            os.path.join(dicom_dir, f)
            for f in os.listdir(dicom_dir)
            if f.endswith(".dcm")
        ]
    )
    if not dicom_files:
        print("No DICOM files found.")
        return

    # Read metadata from the first DICOM file
    reference_dicom = pydicom.dcmread(dicom_files[0])
    pixel_spacing = reference_dicom.PixelSpacing  # (row spacing, col spacing)
    slice_thickness = reference_dicom.SliceThickness
    image_orientation = \
        np.array(reference_dicom.ImageOrientationPatient).reshape(2, 3)
    image_position = np.array(reference_dicom.ImagePositionPatient)

    # Read slices and store shape information
    slices = []
    slice_positions = []
    slice_shapes = []

    for dicom_file in dicom_files:
        dicom_data = pydicom.dcmread(dicom_file)
        if "PixelData" in dicom_data and "ImagePositionPatient" in dicom_data:
            slice_positions.append(dicom_data.ImagePositionPatient[2])
            image_data = dicom_data.pixel_array
            slices.append(image_data)
            slice_shapes.append(image_data.shape)

    # Identify the most common shape among slices
    common_shape = Counter(slice_shapes).most_common(1)[0][0]
    print(f"Most common slice shape: {common_shape}")

    # Filter slices to only include those with the common shape
    filtered_slices = []
    filtered_positions = []
    for i, slice_data in enumerate(slices):
        if slice_data.shape == common_shape:
            filtered_slices.append(slice_data)
            filtered_positions.append(slice_positions[i])
        else:
            print(f"Skipping slice {i} with shape {slice_data.shape}")

    # Ensure we have slices to work with after filtering
    if not filtered_slices:
        print("No slices with a consistent shape found.")
        return

    # Sort filtered slices by z position
    sorted_indices = np.argsort(filtered_positions)
    sorted_slices = [filtered_slices[i] for i in sorted_indices]

    # Convert list of slices to a 3D numpy array
    volume_3d = np.stack(sorted_slices, axis=-1)

    # Define affine transformation matrix
    row_spacing, col_spacing = pixel_spacing
    affine = np.eye(4)
    affine[0, :3] = image_orientation[0] * row_spacing
    affine[1, :3] = image_orientation[1] * col_spacing
    affine[:3, 3] = image_position  # Position of the first slice
    affine[2, 2] = slice_thickness

    # Create a NIfTI image using nibabel and save it
    nifti_image = nib.Nifti1Image(volume_3d, affine)
    nib.save(nifti_image, output_path)
    print(f"NIfTI file saved to {output_path}")
    return volume_3d, affine


def create_binary_mask_from_rtss(
    rtss_path, roi_name_substring, output_path, dicom_folder
):
    """
    Create a binary mask from an RT Structure Set DICOM file.
    """
    rtss = pydicom.dcmread(rtss_path)
    roi_numbers = [
        roi.ROINumber
        for roi in rtss.StructureSetROISequence
        if roi_name_substring.lower() in roi.ROIName.lower()
    ]
    if not roi_numbers:
        print(f"No ROI containing '{roi_name_substring}' found in {rtss_path}")
        return

    dicom_files = [f for f in os.listdir(dicom_folder) if f.endswith(".dcm")]
    slice_positions = []
    reference_dicom = None

    for dicom_file in dicom_files:
        dicom_data = pydicom.dcmread(os.path.join(dicom_folder, dicom_file))
        if "ImagePositionPatient" in dicom_data:
            slice_positions.append(dicom_data.ImagePositionPatient[2])
            if reference_dicom is None:
                reference_dicom = dicom_data

    if not reference_dicom:
        print("No suitable image files found.")
        return

    # Sort slice positions using z-axis and initialize mask
    slice_positions = np.array(slice_positions)
    slice_positions.sort()
    mask_shape = (reference_dicom.Rows, reference_dicom.Columns,
                  len(slice_positions))
    mask = np.zeros(mask_shape, dtype=np.uint8)
    pixel_spacing = np.array(reference_dicom.PixelSpacing)
    image_position = np.array(reference_dicom.ImagePositionPatient)
    image_orientation = np.array(reference_dicom.ImageOrientationPatient)\
        .reshape(2, 3)
    row_dir, col_dir = image_orientation

    # Fill the mask using contours
    for roi_contour in rtss.ROIContourSequence:
        if roi_contour.ReferencedROINumber not in roi_numbers:
            continue
        for contour_seq in roi_contour.ContourSequence:
            contour_data = np.array(contour_seq.ContourData).reshape(-1, 3)
            contour_z = contour_data[0, 2]
            slice_index = np.argmin(np.abs(slice_positions - contour_z))
            if abs(slice_positions[slice_index] - contour_z) > 1.0:
                continue

            pixel_coords = []
            for point in contour_data:
                relative_point = point - image_position
                x_pixel = np.dot(relative_point, row_dir) / pixel_spacing[0]
                y_pixel = np.dot(relative_point, col_dir) / pixel_spacing[1]
                pixel_coords.append([y_pixel, x_pixel])

            pixel_coords = np.round(pixel_coords).astype(int)
            y_coords, x_coords = pixel_coords[:, 0], pixel_coords[:, 1]
            rr, cc = polygon(
                y_coords, x_coords, (reference_dicom.Rows,
                                     reference_dicom.Columns)
            )
            mask[rr, cc, slice_index] = 1

    # Save mask as NIfTI
    mask_affine = np.diag(
        [pixel_spacing[0], pixel_spacing[1], slice_positions[1] 
         - slice_positions[0], 1]
    )
    mask_image = nib.Nifti1Image(mask, mask_affine)
    nib.save(mask_image, output_path)
    print(f"Binary mask saved to {output_path}")
    return mask


def align_mask_with_volume(binary_mask, volume_data, method="pad"):
    """
    Align binary mask with the volume data.
    """
    mask_slices = binary_mask.shape[2]
    volume_slices = volume_data.shape[2]
    if method == "pad":
        if mask_slices < volume_slices:
            padding = (volume_slices - mask_slices) // 2
            mask_padded = np.pad(
                binary_mask,
                ((0, 0), (0, 0), (padding, padding)),
                mode="constant",
                constant_values=0,
            )
            return mask_padded, volume_data
        elif volume_slices < mask_slices:
            padding = (mask_slices - volume_slices) // 2
            volume_padded = np.pad(
                volume_data,
                ((0, 0), (0, 0), (padding, padding)),
                mode="constant",
                constant_values=0,
            )
            return binary_mask, volume_padded
    elif method == "crop":
        if mask_slices > volume_slices:
            start = (mask_slices - volume_slices) // 2
            return binary_mask[:, :, start: start + volume_slices], volume_data
        elif volume_slices > mask_slices:
            start = (volume_slices - mask_slices) // 2
            return binary_mask, volume_data[:, :, start: start + mask_slices]
    return binary_mask, volume_data


def save_aligned_volume_and_mask(
    dicom_dir,
    rtss_path,
    roi_name_substring,
    output_volume_path,
    output_mask_path,
    alignment_method="pad",
):
    volume_data, affine = convert_dicom_to_nifti(dicom_dir, output_volume_path)
    mask = create_binary_mask_from_rtss(
        rtss_path, roi_name_substring, output_mask_path, dicom_dir
    )
    if mask is not None:
        aligned_mask, aligned_volume = align_mask_with_volume(
            mask, volume_data, method=alignment_method
        )
        nib.save(nib.Nifti1Image(aligned_mask, affine), output_mask_path)
        print(f"Aligned mask saved to {output_mask_path}")


def process_all_rtss_files(root_dir, roi_name_substring="vol"):
    """
    Process all folders in the root directory that contain DICOM files and an RT Structure Set (RTSS) file.
    For each folder, generate a 3D volume NIfTI, a binary mask NIfTI, and aligned versions if needed.
    """
    for folder_name in os.listdir(root_dir):
        # Filter folders based on naming convention (modify if needed)
        if folder_name.startswith("vs_gk_") and folder_name.endswith("_t1"):
            folder_path = os.path.join(root_dir, folder_name)
            rtss_path = os.path.join(folder_path, "RTSS.dcm")

            # Verify DICOM files are present
            dicom_files = [f for f in os.listdir(folder_path)
                           if f.endswith(".dcm")]
            if not dicom_files:
                print(f"No DICOM files found in {folder_path}")
                continue

            # Define output paths for NIfTI files
            dicom_output_path = os.path.join(folder_path, "3D_volume.nii")
            output_path_mask = os.path.join(
                folder_path, f"{roi_name_substring}_mask.nii"
            )
            dicom_aligned_output_path = os.path.join(
                folder_path, "3D_aligned_volume.nii"
            )
            output_mask_aligned_path = os.path.join(
                folder_path, f"aligned_{roi_name_substring}_mask.nii"
            )

            # Check if RTSS file exists
            if os.path.exists(rtss_path):
                # Create binary mask from RTSS and save as NIfTI
                mask = create_binary_mask_from_rtss(
                    rtss_path, roi_name_substring, output_path_mask,
                    folder_path
                )
                if mask is None:
                    print(f"No mask created for {folder_name}, skipping alignment.")
                    continue

            # Convert DICOM series to 3D NIFTI volume
            volume_data, affine = convert_dicom_to_nifti(folder_path,
                                                         dicom_output_path)
            if volume_data is None:
                print(f"Failed to create volume for {folder_name}")
                continue

            # Align mask with volume and save aligned NIFTI files
            aligned_mask, aligned_volume = align_mask_with_volume(
                mask, volume_data, method="pad"
            )
            nib.save(nib.Nifti1Image(aligned_volume, affine),
                     dicom_aligned_output_path)
            nib.save(nib.Nifti1Image(aligned_mask, affine),
                     output_mask_aligned_path)

            print(f"Processed {folder_name}")
            print(f"Volume saved to {dicom_output_path}")
            print(f"Aligned volume saved to {dicom_aligned_output_path}")
            print(f"Mask saved to {output_path_mask}")
            print(f"Aligned mask saved to {output_mask_aligned_path}")


# Usage
root_directory = "/Users/yvesgreatti/github/VS_Seg/data2"
process_all_rtss_files(root_directory)