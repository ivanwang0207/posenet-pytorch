import argparse
import json
from pathlib import Path
from typing import Dict, Union
import numpy as np
import pandas as pd
from tools import *

def normalised_relative_pose_errors(
    predicted: np.ndarray,
    actual: np.ndarray,
    seed: int = 0,
) -> Dict[str, float]:
    """Calculate rotation and translation normalised relative pose error for a set predictions against the ground truth.
    The input array columns should correspond to the following: [Easting, Northing, Height, Roll, Pitch, Yaw]

    Args:
        predicted (np.ndarray): Nx6 array of camera pose predictions
        actual (np.ndarray): Nx6 array of camera pose ground truth values

    Returns:
        dict[str, float]: dictionary of error values keyed with "rotation_error" and "translation_error"
    """

    # Convert Euler angles to quaternions
    predicted_quats = euler_to_quaternion(roll=predicted[:, 3], pitch=predicted[:, 4], yaw=predicted[:, 5])
    actual_quats = euler_to_quaternion(roll=actual[:, 3], pitch=actual[:, 4], yaw=actual[:, 5])

    # Generate transformation matrices
    predicted_arr_transforms = transformation_matrix(predicted[:, :3], predicted_quats)
    actual_arr_transforms = transformation_matrix(actual[:, :3], actual_quats)

    # Select frames: (i, j) are adjacent frames in time
    i_predicted = predicted_arr_transforms[:-1, :, :]
    j_predicted = predicted_arr_transforms[1:, :, :]
    i_actual = actual_arr_transforms[:-1, :, :]
    j_actual = actual_arr_transforms[1:, :, :]

    # Relative pose error of 4x4 transformation matrices
    error44 = ominus(ominus(j_predicted, i_predicted), ominus(j_actual, i_actual))

    rot_err = np.mean(compute_angle(error44))
    trans_err = np.mean(compute_distance(error44))

    return {
        "rotation_error": rot_err,
        "translation_error": trans_err,
    }


def main(predicted_path: Union[str, Path], actual_path: Union[str, Path]) -> Dict[int, Dict[str, float]]:
    """Calculate the rotational and translational normalised relative pose error for the Visual Localisation Challenge.

    Args:
        predicted_path (str | Path): Path to predictions CSV file matching submission format
        actual_path (str | Path): Path to ground truth CSV file

    Returns:
        dict[int, Dict[str, float]]: Dictionary of scores for each trajectory
    """
    predicted_df = pd.read_csv(predicted_path, index_col=["Filename", "TrajectoryId", "Timestamp"])
    actual_df = pd.read_csv(actual_path, index_col=["Filename", "TrajectoryId", "Timestamp"])

    # Validate that indices match
    assert predicted_df.index.equals(actual_df.index)
    # Validate that columns are correct
    assert predicted_df.columns.tolist() == ["Easting", "Northing", "Height", "Roll", "Pitch", "Yaw"]
    assert actual_df.columns.tolist() == ["Easting", "Northing", "Height", "Roll", "Pitch", "Yaw"]

    scores = {}
    for trajectory_id in actual_df.index.get_level_values("TrajectoryId").unique():
        predicted = predicted_df.loc[(slice(None), trajectory_id), :].values
        actual = actual_df.loc[(slice(None), trajectory_id), :].values
        scores[trajectory_id] = normalised_relative_pose_errors(predicted, actual)

    return scores


parser = argparse.ArgumentParser(description=main.__doc__.split("\n")[0])
parser.add_argument("predicted_path", help="Path to predictions CSV.")
parser.add_argument("actual_path", help="Path to ground truth CSV.")

if __name__ == "__main__":
    args = parser.parse_args()
    print(
        json.dumps(
            main(
                predicted_path=args.predicted_path,
                actual_path=args.actual_path,
            ),
            indent=2,
        )
    )
