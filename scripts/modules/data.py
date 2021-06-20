#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Utility classes to create PyTorch ``Dataset``s from MDAnalysis trajectories.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import math
import os

import numpy as np
import MDAnalysis
import MDAnalysis.lib
import torch.utils.data

from .functions.geometry import to_batch_atom_3_shape


# =============================================================================
# TRAJECTORY DATASET
# =============================================================================

class TrajectoryDataset(torch.utils.data.Dataset):
    """PyTorch ``Dataset`` wrapping around an MDAnalysis ``Trajectory``.

    The auxiliary information in the trajectory is automatically discovered,
    and it is returned in the sample dictionary, which includes the following
    keys

    - positions: the positions in MDAnalysis units.
    - index (optional): the index in the original dataset.
    - auxiliary_name1 (optional): the name of eventual auxiliary information
    - auxiliary_name2 ...

    Parameters
    ----------
    trajectory : MDAnalysis.Trajectory
        The trajectory, eventually with auxiliary information.
    subsampler : Callable
        A callable that takes the trajectory as input and returns a list
        (or array of int) including all the frame indices to include in
        the database.
    return_batch_index : bool, optional
        If True, the keyword "index" (see above) is included in the sample
        dictionary.

    """
    def __init__(self, trajectory, subsampler=None, return_batch_index=False):
        super().__init__()

        self.trajectory = trajectory
        self.return_batch_index = return_batch_index

        # Subsample.
        self._trajectory_indices = None
        if subsampler is not None:
            self._trajectory_indices = subsampler(trajectory)

        # We don't need the subsampler anymore but we keep
        # it in case the user wants to retrieve the info.
        self._subsampler = subsampler

    def __len__(self):
        if self._trajectory_indices is None:
            return len(self.trajectory)
        return len(self._trajectory_indices)

    def __getitem__(self, item):
        ts = self.get_ts(item)

        # MDAnalysis loads coordinates with np.float32 dtype. We convert
        # it to the default torch dtype and return them in flattened shape.
        positions =  torch.tensor(np.ravel(ts.positions),
                                  dtype=torch.get_default_dtype())

        # Return the configurations and the auxiliary information.
        sample = {'positions': positions}
        for aux_name, aux_info in ts.aux.items():
            sample[aux_name] = torch.tensor(aux_info)

        # Return the item itself if requested.
        if self.return_batch_index:
            sample['index'] = item
        return sample

    @property
    def trajectory_indices(self):
        """Copy of the indices of the subsampled frames or None if all samples from the trajectory are in the dataset."""
        if self._trajectory_indices is not None:
            return self._trajectory_indices.copy()
        return None

    def get_ts(self, item):
        """Return the MDAnalysis ``Timestep`` object for the given item."""
        if self._trajectory_indices is None:
            ts = self.trajectory[item]
        else:
            ts = self.trajectory[self._trajectory_indices[item]]
        return ts

    def iter_as_ts(self):
        """Iterate over MDAnalysis ``Timestep`` objects."""
        if self._trajectory_indices is None:
            return self.trajectory.__iter__()
        else:
            for i in range(len(self)):
                yield self.get_ts(i)

    def save(
            self,
            topology_file_path,
            output_file_path,
            positions=None,
            indices=None,
            **writer_kwargs
    ):
        """Save the dataset into a file.

        Parameters
        ----------
        topology_file_path : str
            The path to the topology file to instantiate the MDAnalysis
            ``Universe`` object.
        output_file_path : str
            The path to the output file. The format will be detected
            automatically from the extension.
        positions : numpy.ndarray, optional
            Optionally, custom positions can be written instead of those
            in the original trajectory. This allows creating output
            files after the coordinates of the original trajectories are
            mapped with the normalizing flow.
        indices : List[int], optional
            A list of indices between 0 and ``len(self)`` to select only
            a portion of the frames.
        **writer_kwargs
            Other keyword arguments to pass to ``MDAnalysis.Writer``.

        """
        universe = MDAnalysis.Universe(topology_file_path)
        universe.trajectory = self.trajectory

        # Handle default arguments.
        if indices is None:
            indices = range(len(self))

        # Make sure the positions array is in the standard shape.
        if positions is not None:
            positions = to_batch_atom_3_shape(positions)

        # Create the directory before saving.
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        with MDAnalysis.Writer(output_file_path, **writer_kwargs) as writer:
            for positions_idx, dataset_idx in enumerate(indices):
                ts = self.get_ts(dataset_idx)

                # Save old positions if we need to write different ones.
                if positions is not None:
                    ts.positions = positions[positions_idx]
                writer.write(universe)


# =============================================================================
# TRAJECTORY SUBSET
# =============================================================================

class TrajectorySubset(torch.utils.data.Subset):
    """A subset of a ``TrajectoryDataset``.

    The class expands ``torch.utils.data.Subset`` to expose the interface
    of the encapsulated ``TrajectoryDataset``. The class supports nested
    ``TrajectorySubset``s objects.

    Parameters
    ----------
    dataset : TrajectoryDataset or TrajectorySubset
        The whole dataset.
    indices : numpy.ndarray
        A list of indices of the ``dataset`` elements forming the subset.

    """

    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)

        # Make sure indices is an array or the subset search won't work.
        if not isinstance(self.indices, np.ndarray):
            self.indices = np.array(self.indices)

    @property
    def trajectory(self):
        return self.dataset.trajectory

    @property
    def trajectory_indices(self):
        # This might create a copy so let's call the property only once.
        trajectory_indices = self.dataset.trajectory_indices
        if trajectory_indices is None:
            return self.indices.copy()
        return trajectory_indices[self.indices]

    @property
    def return_batch_index(self):
        return self.dataset.return_batch_index

    def __getitem__(self, item):
        sample = super().__getitem__(item)

        # Update the index if return_batch_index is True, which is
        # otherwise set to the Trajectory subsample index.
        if self.return_batch_index is True:
            sample['index'] = item

        return sample

    def get_ts(self, item):
        """Return the MDAnalysis ``Timestep`` object of the frame with the given index."""
        item = self.indices[item]
        return self.dataset.get_ts(item)

    def iter_as_ts(self):
        """Iterate over MDAnalysis ``Timestep`` objects."""
        for i in self.indices:
            yield self.dataset.get_ts(i)

    def save(self, *args, **kwargs):
        """Save the dataset into a file."""
        # Resolve the subset of indices.
        indices = kwargs.pop('indices', None)
        if indices is None:
            indices = self.indices
        else:
            indices = self.indices[indices]
        self.dataset.save(*args, indices=indices, **kwargs)


# =============================================================================
# TRAJECTORY SUBSAMPLER
# =============================================================================

def get_time_subsample_indices(
        equilibration_time, stride_time, dt,
        last_sample_time=None, n_frames=None, t0=0.0
):
    """Subsamples the trajectory at a constant time interval after discarding an initial equilibration.

    All the times passed as arguments must have the same unit.

    Parameters
    ----------
    equilibration_time : float
        The initial time of the trajectory to discard.
    stride_time : float
        The time between samples.
    dt : float
        The time between two frames in the trajectory.
    last_sample_time : float, optional
        The last time step to include in the trajectory. Strictly one
        between this and ``n_frames`` must be passed.
    n_frames : int, optional
        The total number of frames to subsample. Strictly one between
        this and ``last_sample_time`` must be passed.
    t0 : float
        The time of the first frame of the trajectory.

    Returns
    -------
    trajectory_indices : numpy.ndarray
        The frame indices selected after subsampling.

    """
    if (n_frames is None) == (last_sample_time is None):
        raise ValueError('One and only one between n_frames and last_sample_time must be passed.')

    # Take into account the time of the initial frame.
    first_sample_time = equilibration_time + stride_time
    first_sample_idx = (first_sample_time - t0) / dt
    stride_idx = stride_time / dt
    if n_frames is None:
        # The +1 is to consider also the frame at time t0.
        n_frames = (last_sample_time - t0) / dt + 1

    # Make sure the options are compatible.
    if not math.isclose(first_sample_idx, round(first_sample_idx), rel_tol=1e-4):
        raise ValueError(f'Trajectory with timestep {dt}ps is not '
                         f'compatible with first sample time {first_sample_time}ps')
    if not math.isclose(stride_idx, round(stride_idx), rel_tol=1e-4):
        raise ValueError(f'Trajectory with timestep {dt}ps is not '
                         f'compatible with sample stride {stride_time}ps')

    first_sample_idx = int(round(first_sample_idx))
    stride_idx = int(round(stride_idx))
    trajectory_indices = np.arange(first_sample_idx, n_frames, stride_idx, dtype=np.int)

    return trajectory_indices


class TrajectorySubsampler:
    """Subsample the trajectory at regular intervals after discarding an initial equilibration.

    This class can be passed as a ``subsampler`` argument in the
    ``TrajectoryDataset`` constructor.

    Parameters
    ----------
    equilibration_time : pint.Quantity
        The initial time of the equilibration to discard.
    stride_time : pint.Quantity
        The time between samples.
    cv_range : Tuple[float] or List[Tuple[float]]
        A pair with the minimum and maximum CV considered. If a list
        of pairs, all frames within any of the CV ranges in the list
        are included.

    """
    def __init__(self, equilibration_time, stride_time, cv_range=None):
        self.equilibration_time = equilibration_time
        self.stride_time = stride_time
        self.cv_range = cv_range

    def __call__(self, trajectory):
        """Return the frame indices that are part of the dataset."""
        # Convert the units into MDAnalysis internal coordinate system.
        equilibration_time = self.equilibration_time.to('ps').magnitude
        stride_time = self.stride_time.to('ps').magnitude

        trajectory_indices = get_time_subsample_indices(
            equilibration_time, stride_time, trajectory.dt,
            n_frames=trajectory.n_frames, t0=trajectory[0].time
        )

        # Filter the samples within the CV range.
        if self.cv_range is not None:
            cv_col_idx = trajectory.get_aux_attribute('plumed', 'get_column_idx')('cv')
            if not isinstance(self.cv_range, list):
                cv_range = [self.cv_range]
            else:
                cv_range = self.cv_range

            indices_to_keep = []
            for i, frame_idx in enumerate(trajectory_indices):
                ts = trajectory[frame_idx]
                cv = ts.aux['plumed'][cv_col_idx]
                for lb, hb in cv_range:
                    if lb <= cv <= hb:
                        indices_to_keep.append(i)
                        break

            trajectory_indices = trajectory_indices[indices_to_keep]

        return trajectory_indices


# =============================================================================
# FRAME TRANSFORMATIONS
# =============================================================================

def _vector_plane_angle(vector, normal):
    """Compute the angle between vector and plane represented by its normal vector."""
    x = np.dot(vector, normal) / (np.linalg.norm(vector) * np.linalg.norm(normal))
    # Catch roundoffs that lead to nan otherwise.
    if x > 1.0:
        return np.pi/2
    elif x < -1.0:
        return -np.pi/2
    return np.arcsin(x)


class RemoveTransRotDOF:
    """A transformation that removes the translational/rotational degrees of freedom.

    This is intended as a Transformation object for an MDAnalysis Trajectory.

    The transformation will move the center of the coordinate system on
    an atom and reorient the axis so that there are always two atoms that
    are constrained to stay on an axis and plane respectively.

    Parameters
    ----------
    center_atom_idx : int, optional
        The index of the atom that is centered in the origin.
    axis_atom_idx : int, optional
        The index of the atom that is forced on the given axis.
    plane_atom_idx : int, optional
        The index of the atom that is forced on the given plane.
    axis : str, optional
        The axis on which the position of ``axis_atom_idx`` is forced.
        This is either 'x', 'y', or 'z'.
    plane : str, optional
        The plane on which the position of ``plane_atom_idx`` is forced.
        This is either 'xy', 'yz', or 'xz'.
    round_off_imprecisions : bool, optional
        As a result of the constrains, several coordinates should be
        exactly 0.0, but numerical errors may cause these to deviate
        from it. Setting this to ``True`` truncate the least significant
        decimal values of the constrained degrees of freedom.

    Attributes
    ----------
    constrained_dof_indices : numpy.ndarray
        An array of the indices of the degrees of freedom that are
        fixed by the transformation. This is lazily initialized only
        after the first time the transformation is called.

    """

    # Conversion string representation to numpy representation.
    _AXIS = {
        'x': np.array([1.0, 0.0, 0.0]),
        'y': np.array([0.0, 1.0, 0.0]),
        'z': np.array([0.0, 0.0, 1.0]),
    }

    def __init__(
            self,
            center_atom_idx=0,
            axis_atom_idx=1,
            plane_atom_idx=2,
            axis='z',
            plane='xz',
            round_off_imprecisions=False
    ):
        if (center_atom_idx == axis_atom_idx or
            axis_atom_idx == plane_atom_idx or
            center_atom_idx == plane_atom_idx):
            raise ValueError('The three constrained atoms must be different.')

        self._center_atom_idx = center_atom_idx
        self._axis_atom_idx = axis_atom_idx
        self._plane_atom_idx = plane_atom_idx

        if axis not in plane:
            raise ValueError('Cannot constrained atom {plane_atom_idx} to stay '
                             'on a plane that does not contain atom {axis_atom_idx}.')

        self.round_off_imprecisions = round_off_imprecisions

        # Save the third axis (not self.axis and not self.plane) which is
        # needed to determine the direction of the rotation to the plane.
        self._third_axis = [x for name, x in self._AXIS.items()
                            if (name not in axis) and (name in plane)][0]

        # Save the axis as a vector.
        self.axis = self._AXIS[axis]

        # We save the plane as its normal vector.
        self.plane = [x for name, x in self._AXIS.items()
                      if name not in plane][0]

        # Determine the degrees of freedom that are fixed by the transformation.
        self._constrained_dof_indices = None
        self._update_constrained_dof_indices()

    @property
    def constrained_dof_indices(self):
        """The indices of the degrees of freedom that are constrained.

        This is None until it the object is executed as a callable by the
        MDAnalysis trajectory. The indices are in flat format (i.e., they
        refer to a single n_atoms*3 array rather than a 2D array of size
        (n_atoms, 3).
        """
        return self._constrained_dof_indices

    def __call__(self, ts):
        # Center the coordinate system on the center atom.
        ts.positions -= ts.positions[self._center_atom_idx]

        # Find the direction perpendicular to the plane formed by the
        # center atom, the axis atom, and the axis.
        rotation_axis = np.cross(ts.positions[self._axis_atom_idx], self.axis)
        vector_vector_angle = MDAnalysis.lib.mdamath.angle(
            ts.positions[self._axis_atom_idx], self.axis
        )
        rotation_matrix_axis = MDAnalysis.lib.transformations.rotation_matrix(
            vector_vector_angle, rotation_axis
        )

        # To bring the plane atom in position, we perform a rotation about
        # self.axis so that we don't modify the position of the axis atom.
        # We perform the first rotation only on the atom position that will
        # determine the next rotation matrix for now so that we run only
        # a single matmul on all atoms.
        plane_atom_position = np.dot(
            rotation_matrix_axis[:3, :3], ts.positions[self._plane_atom_idx]
        )
        # Project the atom on the plane perpendicular to the rotation
        # axis plane to measure the rotation angle.
        plane_atom_position = plane_atom_position - self.axis * np.dot(plane_atom_position, self.axis)
        vector_plane_angle = _vector_plane_angle(
            plane_atom_position, self.plane
        )
        # rotation_clockwise == 1 (rotation is performed clockwise) or -1
        # (rotation needs to be performed counterclockwise).
        rotation_clockwise = -np.sign(np.dot(plane_atom_position, self._third_axis))
        rotation_matrix_plane = MDAnalysis.lib.transformations.rotation_matrix(
            rotation_clockwise * vector_plane_angle, self.axis
        )

        # Now build the rotation composition before applying the transformation.
        rotation_matrix = rotation_matrix_plane @ rotation_matrix_axis
        ts.positions = ts.positions @ (rotation_matrix[:3, :3]).T

        # Now round off numerical imprecisions.
        if self.round_off_imprecisions:
            for constrained_dof_idx in self._constrained_dof_indices:
                atom_idx = constrained_dof_idx // 3
                xyz_idx = constrained_dof_idx % 3
                ts.positions[atom_idx, xyz_idx] = 0.0

        return ts

    def _update_constrained_dof_indices(self):
        """Set self._constrained_dof_indices from the internal state."""
        # The translation fixes all degrees of freedom of the center atom.
        fixed_dofs = self._get_dofs_atoms(self._center_atom_idx)

        # The axis atom has only non-zero coordinates in the self.axis direction.
        atom_dofs = self._get_dofs_atoms(self._axis_atom_idx)
        fixed_indices = np.where(self.axis == 0)[0].tolist()
        fixed_dofs.extend([atom_dofs[i] for i in range(3) if i in fixed_indices])

        # The plane atom has only non-zero coordinate on the plane.
        atom_dofs = self._get_dofs_atoms(self._plane_atom_idx)
        fixed_idx = np.where(self.plane > 0)[0].tolist()[0]
        fixed_dofs.append(atom_dofs[fixed_idx])

        self._constrained_dof_indices = np.array(sorted(fixed_dofs), dtype=np.int)

    @staticmethod
    def _get_dofs_atoms(atom_idx):
        """Return the indices of the dofs corresponding to the atom positions in the flattened coordinates."""
        return [3*atom_idx + i for i in range(3)]
