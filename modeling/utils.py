from mpi4py import MPI
import dolfin as dl
import numpy as np
from hippylib import MultiVector
from pathlib import Path


def root_print(comm: MPI.Comm, *args, **kwargs) -> None:
    """Thin wrapper around print function to only print from root process.
    The flush argument is set to True for convenience. Please run python in unbuffered mode when using MPI.
        ex: mpirun -np 4 python -u my_script.py

    Args:
        comm (MPI.Comm): The MPI communicator.
    """
    if comm.rank == 0:
        print(*args, **kwargs, flush=True)


def report_mesh_info(mesh: dl.Mesh) -> None:
    """Report basic mesh information.

    Args:
        mesh (dl.Mesh): The dolfin mesh object.

    Returns:
        None: Prints mesh information to stdout.
    """
    SEP = "\n" + "#" * 80 + "\n"  # stdout separator

    comm = mesh.mpi_comm()  # get the MPI communicator from the mesh

    # compute mesh information.
    nvertex = comm.allreduce(mesh.num_vertices(), op=MPI.SUM)
    ncell = comm.allreduce(mesh.num_cells(), op=MPI.SUM)
    hmax = comm.allreduce(mesh.hmax(), op=MPI.MAX)
    hmin = comm.allreduce(mesh.hmin(), op=MPI.MIN)
    vol = dl.assemble(1 * dl.dx(mesh))

    # report the information.
    root_print(comm, SEP)
    root_print(comm, "Mesh info:")
    root_print(comm, f"# vertices:\t\t{nvertex}")
    root_print(comm, f"# cells:\t\t{ncell}")
    root_print(comm, f"max cell size (mm):\t{hmax:.2e}")
    root_print(comm, f"min cell size (mm):\t{hmin:.2e}")
    root_print(comm, f"Area (mm^2):\t\t{vol:.2e}")
    root_print(comm, SEP)


def load_mesh(comm, mesh_fpath: str) -> None:
    """Load a dolfin mesh from file.

    Args:
        comm (dl.MPI.comm_world): MPI communicator.
        mesh_fpath (str): Path to the mesh file.
    """
    suffix = Path(mesh_fpath).suffix

    mesh = dl.Mesh(comm)

    if suffix == ".h5":
        with dl.HDF5File(mesh.mpi_comm(), mesh_fpath, "r") as fid:
            fid.read(mesh, "/mesh", False)
    elif suffix == ".xdmf":
        with dl.XDMFFile(mesh.mpi_comm(), mesh_fpath) as fid:
            fid.read(mesh)
    else:
        raise ValueError(f"Unknown mesh file format: {suffix}")

    return mesh


def mv_to_dense_local(multivector):
    """
    This function converts a MultiVector object to a numpy array
        - :code:`multivector` - hippylib MultiVector object
    """
    multivector_shape = (multivector[0].get_local().shape[0], multivector.nvec())
    out_array = np.zeros(multivector_shape)
    for i in range(multivector_shape[-1]):
        out_array[:, i] = multivector[i].get_local()

    return out_array


def mv_to_dense(multivector):
    """
    This function converts a MultiVector object to a numpy array
        - :code:`multivector` - hippylib MultiVector object
    """
    multivector_shape = (multivector[0].gather_on_zero().shape[0], multivector.nvec())
    out_array = np.zeros(multivector_shape)
    for i in range(multivector_shape[-1]):
        out_array[:, i] = multivector[i].gather_on_zero()

    return out_array


def dense_to_mv_local(dense_array, dl_vector):
    """
    This function converts a numpy array to a MultiVector
        - :code:`dense_array` - numpy array to be transformed
        - :code:`dl_vector` - type :code:`dolfin.Vector` object to be used in the
            MultiVector object constructor
    """
    # This function actually makes no sense
    temp = MultiVector(dl_vector, dense_array.shape[-1])
    for i in range(temp.nvec()):
        temp[i].set_local(dense_array[:, i])
    return temp


def allocate_samples_per_proc(sample_size: int, comm_sampler: MPI.Comm) -> list:
    """Allocate the sample sizes across all processes in a communicator.
    This function divides the total sample size into nearly equal parts for each process,
    distributing any remainder evenly among the first few processes.

    adapted from SOUPy: https://github.com/hippylib/soupy/blob/main/soupy/collectives/mpiUtils.py

    Args:
        sample_size (int): Total number of samples to be allocated across processes.
        comm_sampler (MPI.Comm): MPI communicator for sample parallelism.

    Returns:
        list: A list of integers where each element represents the number of samples allocated to the corresponding process.
    """
    n, r = divmod(sample_size, comm_sampler.size)
    sample_size_allprocs = []
    for i_rank in range(comm_sampler.size):
        if i_rank < r:
            sample_size_allprocs.append(n + 1)
        else:
            sample_size_allprocs.append(n)
    return sample_size_allprocs


class NullCollective:
    """
    No-overhead "Parallel" reduction utilities when a serial system of PDEs is solved on 1 process.

    adapted from SOUPy https://github.com/hippylib/soupy/blob/main/soupy/collectives/collective.py
    """

    def bcast(self, v, root=0):
        return v

    def size(self):
        return 1

    def rank(self):
        return 0

    def allReduce(self, v, op):
        if op.lower() not in ["sum", "avg"]:
            err_msg = "Unknown operation *{0}* in NullCollective.allReduce".format(op)
            raise NotImplementedError(err_msg)

        return v


class ParallelPDECollective:
    """
    Parallel reduction utilities when several serial systems of PDEs (one per process) are solved concurrently.

    adapted from SOUPy: https://github.com/hippylib/soupy/blob/main/soupy/collectives/collective.py
    """

    def __init__(self, comm: MPI.Comm, is_serial_check: bool = False):
        """Constructor.

        Args:
            comm (MPI.Comm): The MPI communicator for parallel processes.
            is_serial_check (bool, optional): Check if the communicator is serial (only one process). Defaults to False.
        """
        self.comm = comm
        self.is_serial_check = is_serial_check

    def size(self):
        return self.comm.Get_size()

    def rank(self):
        return self.comm.Get_rank()

    def _allReduce_array(self, v, op):
        err_msg = "Unknown operation *{0}* in ParallelPDECollective.allReduce".format(op)
        receive = np.zeros_like(v)
        self.comm.Allreduce(v, receive, op=MPI.SUM)
        if op == "sum":
            v[:] = receive
        elif op == "avg":
            v[:] = (1.0 / float(self.size())) * receive
        else:
            raise NotImplementedError(err_msg)
        return v

    def allReduce(self, v, op: str):
        """Reduce a value or array across all processes using the specified operation.

        Cases handled:
            - `v` is a scalar (`float`, `int`);
            - `v` is a numpy array (NOTE: `v` will be overwritten)
            - `v` is a  :code:`dolfin.Vector` (NOTE: `v` will be overwritten)

        Args:
            v: Data to be reduced. It can be a scalar, numpy array, or a dolfin.Vector.
            op (str): The operation to perform for reduction. Can be "sum" or "avg" (case insensitive).

        Raises:
            NotImplementedError: If the type of `v` is not supported for reduction or if the operation is not recognized.

        Returns:
            Reduced value or array: The same type as `v`, now reduced across all processes.
        """
        op = op.lower()

        if type(v) in [float, np.float64]:
            v_array = np.array([v], dtype=np.float64)
            self._allReduce_array(v_array, op)
            return v_array[0]

        elif type(v) in [int, np.int32]:
            v_array = np.array([v], dtype=np.int32)
            self._allReduce_array(v_array, op)
            return v_array[0]

        elif (type(v) is np.array) or (type(v) is np.ndarray):
            return self._allReduce_array(v, op)

        elif hasattr(v, "mpi_comm") and hasattr(v, "get_local"):
            # v is most likely a dl.Vector
            if self.is_serial_check:
                assert v.mpi_comm().Get_size() == 1
            v_array = v.get_local()
            self._allReduce_array(v_array, op)
            v.set_local(v_array)
            v.apply("")

            return v
        elif hasattr(v, "nvec"):
            for i in range(v.nvec()):
                self.allReduce(v[i], op)
            return v
        else:
            if self.is_serial_check:
                msg = "ParallelPDECollective.allReduce not implement for v of type {0}".format(type(v))
            else:
                msg = "ParallelPDECollective.allReduce not implement for v of type {0}".format(type(v))
            raise NotImplementedError(msg)

    def bcast(self, v, root=0):
        """Broadcast a value or array from the root process to all other processes.

        Cases handled:
            - `v` is a scalar (`float`, `int`);
            - `v` is a numpy array (NOTE: `v` will be overwritten)
            - `v` is a  :code:`dolfin.Vector` (NOTE: `v` will be overwritten)

        Args:
            v: Data to be broadcasted. It can be a scalar, numpy array, or a dolfin.Vector.
            root (`int`, optional): Proccess within communicator where data to be broadcasted lives. Defaults to 0.

        Raises:
            NotImplementedError: If the type of `v` is not supported for broadcasting.

        Returns:
            Broadcasted value or array: The same type as `v`, now available on all processes.
        """

        if type(v) in [float, np.float64, int, np.int32]:
            v_array = np.array([v])
            self.comm.Bcast(v_array, root=root)
            return v_array[0]

        if type(v) in [np.array, np.ndarray]:
            self.comm.Bcast(v, root=root)
            return v

        elif hasattr(v, "mpi_comm") and hasattr(v, "get_local"):
            # v is most likely a dl.Vector
            if self.is_serial_check:
                assert v.mpi_comm().Get_size() == 1

            v_local = v.get_local()
            self.comm.Bcast(v_local, root=root)
            v.set_local(v_local)
            v.apply("")

            return v
        elif hasattr(v, "nvec"):
            for i in range(v.nvec()):
                self.bcast(v[i], root=root)
            return v

        else:
            if self.is_serial_check:
                msg = "ParallelPDECollective.bcast not implement for v of type {0}".format(type(v))
            else:
                msg = "ParallelPDECollective.bcast not implement for v of type {0}".format(type(v))
            raise NotImplementedError(msg)


def MultipleSerialPDECollective(comm):
    return ParallelPDECollective(comm, is_serial_check=True)
