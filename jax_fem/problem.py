import os

import numpy as onp
import jax
import jax.numpy as np
import jax.flatten_util
from dataclasses import dataclass
import functools
from typing import Any, ClassVar, Optional

from jax_fem.utils import timeit 
from jax_fem.generate_mesh import Mesh
from jax_fem.fe import FiniteElement
from jax_fem import logger


# cudss-plus 2026-11-04 (options A+B of the stage-cost study): host path of the element tangents.
# The original compute_newton_vars copied every Newton tangent three times on the host: onp.vstack of pageable
# device->host copies of the chunks, onp.array of the result, and onp.hstack once per face block (0.94 GB each for
# the 0119 mechanics). Here every chunk and face block is copied once, in order, into one host buffer of len(I)
# values that the problem keeps and reuses, through pinned host memory when the backend has it. The values and
# their order are unchanged. Problem.V is that buffer; its consumers (the PETSc tangent cache's setValuesCOO, the
# scipy COO assembly) copy it before the next build. JAX_FEM_TANGENT_HOST_PATH=legacy restores the original copies.
TANGENT_HOST_PATH = os.environ.get("JAX_FEM_TANGENT_HOST_PATH", "pinned").strip().lower()
_PINNED_SHARDINGS = {}


def _pinned_host_sharding(array):
    """SingleDeviceSharding(memory_kind="pinned_host") for the device of ``array``; None on the CPU backend or
    when the backend has no pinned host memory (then the copies are pageable)."""
    try:
        dev = next(iter(array.devices()))
    except Exception:
        return None
    if dev.platform == "cpu":
        return None
    key = (dev.platform, dev.id)
    if key not in _PINNED_SHARDINGS:
        try:
            from jax.sharding import SingleDeviceSharding
            sharding = SingleDeviceSharding(dev, memory_kind="pinned_host")
            jax.device_put(np.zeros((1,), dtype=array.dtype), sharding).block_until_ready()
            _PINNED_SHARDINGS[key] = sharding
        except Exception as exc:
            logger.warning(f"pinned_host transfers unavailable ({exc}); element tangents use pageable copies")
            _PINNED_SHARDINGS[key] = None
    return _PINNED_SHARDINGS[key]


# cudss-plus 2026-11-04 (dispatch fix, prior P4): the chunk loop of split_and_compute_cell sliced every input
# eagerly (17 inputs x 7 chunks = 119 slice dispatches, ~0.1 s per 0119 mechanics build while the jitted kernel
# call returns in 0.2 ms). One jitted dynamic_slice call per chunk cuts the same rows. JAX_FEM_CHUNK_SLICE=eager
# restores the per-array slicing.
CHUNK_SLICE = os.environ.get("JAX_FEM_CHUNK_SLICE", "jit").strip().lower()


@functools.partial(jax.jit, static_argnums=(1,))
def _slice_rows(arrays, size, start):
    return [jax.lax.dynamic_slice_in_dim(x, start, size, axis=0) for x in arrays]


def _cell_chunk(input_collection, i, batch_size, num_cuts, num_cells):
    """Inputs of chunk ``i``: rows [i*batch_size, (i+1)*batch_size), the last chunk running to the end."""
    start = i * batch_size
    if CHUNK_SLICE != "eager":
        leaves, treedef = jax.tree_util.tree_flatten(input_collection)
        if all(getattr(x, "ndim", 0) >= 1 and x.shape[0] == num_cells for x in leaves):
            stop = (i + 1) * batch_size if i < num_cuts - 1 else num_cells
            return jax.tree_util.tree_unflatten(treedef, _slice_rows(leaves, stop - start, start))
    if i < num_cuts - 1:
        return jax.tree_util.tree_map(lambda x: x[start:start + batch_size], input_collection)
    return jax.tree_util.tree_map(lambda x: x[start:], input_collection)


class _TangentHostWriter:
    """``np_version`` stand-in for split_and_compute_cell(jac_flag=True) in compute_newton_vars.

    ``vstack`` of 3-D chunks (element tangents) copies them, in order, into the problem's reused host buffer and
    returns a view of that part; ``vstack`` of 2-D chunks (cell residuals) concatenates on the device, where the
    residual is scattered next. ``write`` appends further blocks (the face tangents). Other attributes are numpy's.
    """

    def __init__(self, problem):
        self.problem = problem
        self.out = None
        self.offset = 0
        self.wrote_cells = False

    def _buffer(self, dtype):
        n = int(self.problem.I.shape[0])
        buf = getattr(self.problem, "_tangent_host_buffer", None)
        if buf is None or buf.shape[0] != n or buf.dtype != dtype:
            buf = onp.empty(n, dtype=dtype)
            self.problem._tangent_host_buffer = buf
        return buf

    def write(self, arrays):
        """Copy ``arrays`` (device or host, any shape) one after another into the buffer; returns (start, stop)."""
        arrays = list(arrays)
        start = self.offset
        if not arrays:
            return start, start
        if self.out is None:
            self.out = self._buffer(onp.dtype(arrays[0].dtype))
        sizes = [int(onp.prod(a.shape)) for a in arrays]
        if start + sum(sizes) > self.out.shape[0]:
            raise ValueError("element tangent blocks exceed len(Problem.I)")
        pinned = None
        for a in arrays:
            if isinstance(a, jax.Array):
                pinned = _pinned_host_sharding(a)
                break

        def begin(a):
            if pinned is not None and isinstance(a, jax.Array):
                return jax.device_put(a, pinned)
            return a

        # two deep: the next block's device->host copy runs while this one is copied into the buffer
        pending = begin(arrays[0])
        for i, size in enumerate(sizes):
            nxt = begin(arrays[i + 1]) if i + 1 < len(arrays) else None
            self.out[self.offset:self.offset + size] = onp.asarray(pending).reshape(-1)
            self.offset += size
            pending = nxt
        return start, self.offset

    def vstack(self, arrays):
        arrays = list(arrays)
        if arrays and getattr(arrays[0], "ndim", 0) == 3:
            start, stop = self.write(arrays)
            self.wrote_cells = True
            rows = sum(int(a.shape[0]) for a in arrays)
            return self.out[start:stop].reshape((rows,) + tuple(arrays[0].shape[1:]))
        return np.concatenate(arrays, axis=0)

    def finish(self):
        if self.out is None or self.offset != self.out.shape[0]:
            raise ValueError(
                f"element tangents filled {self.offset} of {int(self.problem.I.shape[0])} COO values"
            )
        return self.out

    def __getattr__(self, name):
        return getattr(onp, name)



@dataclass
class Problem:
    """Problem class to handle one FE variable or multiple coupled FE variables.

    Attributes
    ----------
    mesh : Mesh
        :attr:`~jax_fem.fe.FiniteElement.mesh`
    vec : int
        :attr:`~jax_fem.fe.FiniteElement.vec`
    dim : int
        :attr:`~jax_fem.fe.FiniteElement.dim`
    ele_type : str
        :attr:`~jax_fem.fe.FiniteElement.ele_type`
    quadrature_rule
        :attr:`~jax_fem.fe.FiniteElement.quadrature_rule`
    quadrature_order : int
        :attr:`~jax_fem.fe.FiniteElement.quadrature_order`
    dirichlet_bc_info : list
        :attr:`~jax_fem.fe.FiniteElement.dirichlet_bc_info`
    location_fns : list
        A list of location functions useful for surface integrals in the weak form.
        Such surface integral can be related to Neumann boundary condition, or an integral contributing to the stiffness matrix.
        Each callable takes a point (NumpyArray) and returns a boolean indicating if the point satisfies the location condition.
        For example, ::
        
            [lambda point: np.isclose(point[0], 0., atol=1e-5)]

    additional_info: tuple
        Any other information that might be useful can be stored here. This is problem dependent.
    """
    mesh: Mesh
    vec: int
    dim: int
    ele_type: str = 'HEX8'
    quadrature_rule: Any = None
    quadrature_order: int = None
    dirichlet_bc_info: list = None
    location_fns: list = None
    additional_info: tuple = ()
    cell_assembly_num_cuts: ClassVar[int] = 20
    cell_assembly_target_batch_size: ClassVar[Optional[int]] = None

    def __post_init__(self):

        if type(self.mesh) != type([]):
            self.mesh = [self.mesh]
            self.vec = [self.vec]
            self.ele_type = [self.ele_type]
            self.quadrature_rule = [self.quadrature_rule]
            self.quadrature_order = [self.quadrature_order]
            self.dirichlet_bc_info = [self.dirichlet_bc_info]

        self.num_vars = len(self.mesh)

        self.fes = [FiniteElement(mesh=self.mesh[i], 
                                  vec=self.vec[i], 
                                  dim=self.dim, 
                                  ele_type=self.ele_type[i], 
                                  quadrature_rule=self.quadrature_rule[i] if type(self.quadrature_rule) == type([]) else self.quadrature_rule,
                                  quadrature_order=self.quadrature_order[i] if type(self.quadrature_order) == type([]) else self.quadrature_order,
                                  dirichlet_bc_info=self.dirichlet_bc_info[i] if type(self.dirichlet_bc_info) == type([]) else self.dirichlet_bc_info) \
                    for i in range(self.num_vars)] 

        self.cells_list = [fe.cells for fe in self.fes]
        # Assume all fes have the same number of cells, same dimension
        self.num_cells = self.fes[0].num_cells
        self.boundary_inds_list = self.fes[0].get_boundary_conditions_inds(self.location_fns)

        self.offset = [0] 
        for i in range(len(self.fes) - 1):
            self.offset.append(self.offset[i] + self.fes[i].num_total_dofs)

        # (num_cells, num_nodes*vec + ...)
        inds = self._cell_dof_indices(self.cells_list)
        self.I = onp.repeat(inds[:, :, None], inds.shape[1], axis=2).reshape(-1)
        self.J = onp.repeat(inds[:, None, :], inds.shape[1], axis=1).reshape(-1)
        self.cells_list_face_list = []

        for i, boundary_inds in enumerate(self.boundary_inds_list):
            cells_list_face = [cells[boundary_inds[:, 0]] for cells in self.cells_list] # [(num_selected_faces, num_nodes), ...]
            inds_face = self._cell_dof_indices(cells_list_face) # (num_selected_faces, num_nodes*vec + ...)
            I_face = onp.repeat(inds_face[:, :, None], inds_face.shape[1], axis=2).reshape(-1)
            J_face = onp.repeat(inds_face[:, None, :], inds_face.shape[1], axis=1).reshape(-1)
            self.I = onp.hstack((self.I, I_face))
            self.J = onp.hstack((self.J, J_face))
            self.cells_list_face_list.append(cells_list_face)

        self.cells_flat = self._flatten_cells(self.cells_list) # (num_cells, num_nodes + ...)
        self._init_unflatten_fns()

        self.num_nodes_cumsum = onp.cumsum([0] + [fe.num_nodes for fe in self.fes])

        self.initialize_geometric_quantities()

        self.internal_vars = ()
        self.internal_vars_surfaces = [() for _ in range(len(self.boundary_inds_list))]
        self.custom_init(*self.additional_info)
        self.pre_jit_fns()

    def initialize_geometric_quantities(self, fes_points=None):
        """Initialize geometric quantities for the problem.
        """
        fes_points = [fe.points for fe in self.fes] if fes_points is None else fes_points

        for i, fe in enumerate(self.fes):
            # (num_cells, num_quads, num_nodes, dim), (num_cells, num_quads)
            fe.shape_grads, fe.JxW = fe.get_shape_grads(fes_points[i])
            # (num_cells, num_quads, num_nodes, 1, dim)
            fe.v_grads_JxW = fe.shape_grads[:, :, :, None, :] * fe.JxW[:, :, None, None, None]

        # Use jax.numpy so differentiable nodal coords (JAX arrays) are not cast to NumPy here.
        # (num_cells, num_vars, num_quads)
        self.JxW = np.transpose(np.stack([fe.JxW for fe in self.fes]), axes=(1, 0, 2))
        # (num_cells, num_quads, num_nodes +..., dim)
        self.shape_grads = np.concatenate([fe.shape_grads for fe in self.fes], axis=2)
        # (num_cells, num_quads, num_nodes + ..., 1, dim)
        self.v_grads_JxW = np.concatenate([fe.v_grads_JxW for fe in self.fes], axis=2)

        # TODO: Now assumes all vars share the same quad points
        # (num_cells, num_quads, dim)
        self.physical_quad_points = self.fes[0].get_physical_quad_points(fes_points[0])  

        self.selected_face_shape_grads = []
        self.nanson_scale = []
        self.selected_face_shape_vals = []
        self.physical_surface_quad_points = []
        for boundary_inds in self.boundary_inds_list:
            s_shape_grads = []
            n_scale = []
            s_shape_vals = []
            for i, fe in enumerate(self.fes):
                # (num_selected_faces, num_face_quads, num_nodes, dim), (num_selected_faces, num_face_quads)
                face_shape_grads_physical, nanson_scale = fe.get_face_shape_grads(boundary_inds, fes_points[i])  
                selected_face_shape_vals = fe.face_shape_vals[boundary_inds[:, 1]]  # (num_selected_faces, num_face_quads, num_nodes)
                s_shape_grads.append(face_shape_grads_physical)
                n_scale.append(nanson_scale)
                s_shape_vals.append(selected_face_shape_vals)

            # (num_selected_faces, num_face_quads, num_nodes + ..., dim)
            s_shape_grads = np.concatenate(s_shape_grads, axis=2)
            # (num_selected_faces, num_vars, num_face_quads)
            n_scale = np.transpose(np.stack(n_scale), axes=(1, 0, 2))
            # (num_selected_faces, num_face_quads, num_nodes + ...)
            s_shape_vals = np.concatenate(s_shape_vals, axis=2)
            # (num_selected_faces, num_face_quads, dim)
            physical_surface_quad_points = self.fes[0].get_physical_surface_quad_points(boundary_inds, fes_points[0]) 

            self.selected_face_shape_grads.append(s_shape_grads)
            self.nanson_scale.append(n_scale)
            self.selected_face_shape_vals.append(s_shape_vals)
            # TODO: Now assumes all vars share the same quad points
            self.physical_surface_quad_points.append(physical_surface_quad_points)

    def custom_init(self):
        """Child class should override if more things need to be done in initialization
        """
        pass

    def get_laplace_kernel(self, tensor_map):

        def laplace_kernel(cell_sol_flat, cell_shape_grads, cell_v_grads_JxW, *cell_internal_vars):
            # cell_sol_flat: (num_nodes*vec + ...,)
            # cell_sol_list: [(num_nodes, vec), ...]
            # cell_shape_grads: (num_quads, num_nodes + ..., dim)
            # cell_v_grads_JxW: (num_quads, num_nodes + ..., 1, dim)

            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_shape_grads = cell_shape_grads[:, :self.fes[0].num_nodes, :]
            cell_sol = cell_sol_list[0]
            cell_v_grads_JxW = cell_v_grads_JxW[:, :self.fes[0].num_nodes, :, :]
            vec = self.fes[0].vec

            # (1, num_nodes, vec, 1) * (num_quads, num_nodes, 1, dim) -> (num_quads, num_nodes, vec, dim)
            u_grads = cell_sol[None, :, :, None] * cell_shape_grads[:, :, None, :]
            u_grads = np.sum(u_grads, axis=1)  # (num_quads, vec, dim)
            u_grads_reshape = u_grads.reshape(-1, vec, self.dim)  # (num_quads, vec, dim)
            # (num_quads, vec, dim)
            u_physics = jax.vmap(tensor_map)(u_grads_reshape, *cell_internal_vars).reshape(u_grads.shape)
            # (num_quads, num_nodes, vec, dim) -> (num_nodes, vec)
            val = np.sum(u_physics[:, None, :, :] * cell_v_grads_JxW, axis=(0, -1))
            val = jax.flatten_util.ravel_pytree(val)[0] # (num_nodes*vec + ...,)
            return val

        return laplace_kernel

    def get_mass_kernel(self, mass_map):

        def mass_kernel(cell_sol_flat, x, cell_JxW, *cell_internal_vars):
            # cell_sol_flat: (num_nodes*vec + ...,)
            # cell_sol_list: [(num_nodes, vec), ...]
            # x: (num_quads, dim)
            # cell_JxW: (num_vars, num_quads)

            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_sol = cell_sol_list[0]
            cell_JxW = cell_JxW[0]
            vec = self.fes[0].vec
            # (1, num_nodes, vec) * (num_quads, num_nodes, 1) -> (num_quads, num_nodes, vec) -> (num_quads, vec)
            u = np.sum(cell_sol[None, :, :] * self.fes[0].shape_vals[:, :, None], axis=1)
            u_physics = jax.vmap(mass_map)(u, x, *cell_internal_vars)  # (num_quads, vec)
            # (num_quads, 1, vec) * (num_quads, num_nodes, 1) * (num_quads, 1, 1) -> (num_nodes, vec)
            val = np.sum(u_physics[:, None, :] * self.fes[0].shape_vals[:, :, None] * cell_JxW[:, None, None], axis=0)
            val = jax.flatten_util.ravel_pytree(val)[0] # (num_nodes*vec + ...,)
            return val

        return mass_kernel

    def get_surface_kernel(self, surface_map):

        def surface_kernel(cell_sol_flat, x, face_shape_vals, face_shape_grads, face_nanson_scale, *cell_internal_vars_surface):
            # face_shape_vals: (num_face_quads, num_nodes + ...)
            # face_shape_grads: (num_face_quads, num_nodes + ..., dim)
            # x: (num_face_quads, dim)
            # face_nanson_scale: (num_vars, num_face_quads)

            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_sol = cell_sol_list[0]
            face_shape_vals = face_shape_vals[:, :self.fes[0].num_nodes]
            face_nanson_scale = face_nanson_scale[0]

            # (1, num_nodes, vec) * (num_face_quads, num_nodes, 1) -> (num_face_quads, vec)
            u = np.sum(cell_sol[None, :, :] * face_shape_vals[:, :, None], axis=1)
            u_physics = jax.vmap(surface_map)(u, x, *cell_internal_vars_surface)  # (num_face_quads, vec)
            # (num_face_quads, 1, vec) * (num_face_quads, num_nodes, 1) * (num_face_quads, 1, 1) -> (num_nodes, vec)
            val = np.sum(u_physics[:, None, :] * face_shape_vals[:, :, None] * face_nanson_scale[:, None, None], axis=0)

            return jax.flatten_util.ravel_pytree(val)[0]

        return surface_kernel

    def pre_jit_fns(self):
        def value_and_jacfwd(f, x):
            pushfwd = functools.partial(jax.jvp, f, (x, ))
            basis = np.eye(len(x.reshape(-1)), dtype=x.dtype).reshape(-1, *x.shape)
            y, jac = jax.vmap(pushfwd, out_axes=(None, -1))((basis, ))
            return y, jac

        def get_kernel_fn_cell():
            def kernel(cell_sol_flat, physical_quad_points, cell_shape_grads, cell_JxW, cell_v_grads_JxW, *cell_internal_vars):
                """
                universal_kernel should be able to cover all situations (including mass_kernel and laplace_kernel).
                mass_kernel and laplace_kernel are from legacy JAX-FEM. They can still be used, but not mandatory.
                """

                # TODO: If there is no kernel map, returning 0. is not a good choice. 
                # Return a zero array with proper shape will be better.
                if hasattr(self, 'get_mass_map'):
                    mass_kernel = self.get_mass_kernel(self.get_mass_map())
                    mass_val = mass_kernel(cell_sol_flat, physical_quad_points, cell_JxW, *cell_internal_vars)
                else:
                    mass_val = 0.

                if hasattr(self, 'get_tensor_map'):
                    laplace_kernel = self.get_laplace_kernel(self.get_tensor_map())
                    laplace_val = laplace_kernel(cell_sol_flat, cell_shape_grads, cell_v_grads_JxW, *cell_internal_vars)
                else:
                    laplace_val = 0.

                if hasattr(self, 'get_universal_kernel'):
                    universal_kernel = self.get_universal_kernel()
                    universal_val = universal_kernel(cell_sol_flat, physical_quad_points, cell_shape_grads, cell_JxW, 
                        cell_v_grads_JxW, *cell_internal_vars)
                else:
                    universal_val = 0.

                return laplace_val + mass_val + universal_val


            def kernel_jac(cell_sol_flat, *args):
                kernel_partial = lambda cell_sol_flat: kernel(cell_sol_flat, *args)
                return value_and_jacfwd(kernel_partial, cell_sol_flat)  # kernel(cell_sol_flat, *args), jax.jacfwd(kernel)(cell_sol_flat, *args)

            return kernel, kernel_jac

        def get_kernel_fn_face(ind):
            def kernel(cell_sol_flat, physical_surface_quad_points, face_shape_vals, face_shape_grads, face_nanson_scale, *cell_internal_vars_surface):
                """
                universal_kernel should be able to cover all situations (including surface_kernel).
                surface_kernel is from legacy JAX-FEM. It can still be used, but not mandatory.
                """
                if hasattr(self, 'get_surface_maps'):
                    surface_kernel = self.get_surface_kernel(self.get_surface_maps()[ind])
                    surface_val = surface_kernel(cell_sol_flat, physical_surface_quad_points, face_shape_vals,
                        face_shape_grads, face_nanson_scale, *cell_internal_vars_surface)
                else:
                    surface_val = 0.

                if hasattr(self, 'get_universal_kernels_surface'):
                    universal_kernel = self.get_universal_kernels_surface()[ind]
                    universal_val = universal_kernel(cell_sol_flat, physical_surface_quad_points, face_shape_vals,
                        face_shape_grads, face_nanson_scale, *cell_internal_vars_surface)
                else:
                    universal_val = 0.

                return surface_val + universal_val

            def kernel_jac(cell_sol_flat, *args):
                # return jax.jacfwd(kernel)(cell_sol_flat, *args)
                kernel_partial = lambda cell_sol_flat: kernel(cell_sol_flat, *args)
                return value_and_jacfwd(kernel_partial, cell_sol_flat)  # kernel(cell_sol_flat, *args), jax.jacfwd(kernel)(cell_sol_flat, *args)

            return kernel, kernel_jac

        kernel, kernel_jac = get_kernel_fn_cell()
        kernel = jax.jit(jax.vmap(kernel))
        kernel_jac = jax.jit(jax.vmap(kernel_jac))
        self.kernel = kernel
        self.kernel_jac = kernel_jac

        num_surfaces = len(self.boundary_inds_list)
        if hasattr(self, 'get_surface_maps'):
            assert num_surfaces == len(self.get_surface_maps())
        elif hasattr(self, 'get_universal_kernels_surface'):
            assert num_surfaces == len(self.get_universal_kernels_surface()) 
        else:
            assert num_surfaces == 0, "Missing definitions for surface integral"
            

        self.kernel_face = []
        self.kernel_jac_face = []
        for i in range(len(self.boundary_inds_list)):
            kernel_face, kernel_jac_face = get_kernel_fn_face(i)
            kernel_face = jax.jit(jax.vmap(kernel_face))
            kernel_jac_face = jax.jit(jax.vmap(kernel_jac_face))
            self.kernel_face.append(kernel_face)
            self.kernel_jac_face.append(kernel_jac_face)

    def _cell_assembly_cut_count(self):
        """Return the number of chunks used for cell local assembly."""
        target_batch_size = getattr(
            self,
            "cell_assembly_target_batch_size",
            type(self).cell_assembly_target_batch_size,
        )
        if target_batch_size is not None:
            target_batch_size = int(target_batch_size)
            if target_batch_size <= 0:
                raise ValueError(
                    "cell_assembly_target_batch_size must be a positive integer"
                )
            num_cuts = (
                (int(self.num_cells) + target_batch_size - 1)
                // target_batch_size
            )
            return min(max(1, num_cuts), self.num_cells)

        raw_num_cuts = getattr(
            self,
            "cell_assembly_num_cuts",
            type(self).cell_assembly_num_cuts,
        )
        num_cuts = int(raw_num_cuts)
        if num_cuts <= 0:
            raise ValueError("cell_assembly_num_cuts must be a positive integer")
        return min(num_cuts, self.num_cells)

    def _flatten_cells(self, cells_list):
        """Flatten per-cell connectivity arrays."""
        if self.num_vars == 1:
            cells = cells_list[0]
            cell_width = int(onp.prod(cells.shape[1:])) if len(cells.shape) > 1 else 1
            return np.reshape(cells, (self.num_cells, cell_width))
        return jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(
            *cells_list
        )

    def _cell_dof_indices(self, cells_list):
        """Return global DOF indices for each cell or selected face cell."""
        if self.num_vars == 1:
            cells = onp.asarray(cells_list[0])
            vec = int(self.fes[0].vec)
            offset = int(self.offset[0])
            dof_inds = (
                cells[..., None] * vec
                + onp.arange(vec, dtype=cells.dtype)[None, None, :]
                + offset
            )
            cell_width = (cells.shape[1] if cells.ndim > 1 else 1) * vec
            return dof_inds.reshape((cells.shape[0], cell_width))

        def find_ind(*x):
            inds = []
            for i in range(len(x)):
                crt_ind = (
                    self.fes[i].vec * x[i][:, None]
                    + np.arange(self.fes[i].vec)[None, :]
                    + self.offset[i]
                )
                inds.append(crt_ind.reshape(-1))

            return np.hstack(inds)

        return onp.array(jax.vmap(find_ind)(*cells_list))

    def _init_unflatten_fns(self):
        """Initialize flat DOF unflatten helpers."""
        if self.num_vars == 1:
            fe = self.fes[0]
            cell_shape = (int(fe.num_nodes), int(fe.vec))
            sol_shape = (int(fe.num_total_nodes), int(fe.vec))

            def unflatten_dof(flat):
                return [np.reshape(flat, cell_shape)]

            def unflatten_sol_list(flat):
                return [np.reshape(flat, sol_shape)]

            self.unflatten_fn_dof = unflatten_dof
            self.unflatten_fn_sol_list = unflatten_sol_list
            self.num_total_dofs_all_vars = sol_shape[0] * sol_shape[1]
            return

        dumb_array_dof = [np.zeros((fe.num_nodes, fe.vec)) for fe in self.fes]
        _, self.unflatten_fn_dof = jax.flatten_util.ravel_pytree(dumb_array_dof)

        dumb_sol_list = [np.zeros((fe.num_total_nodes, fe.vec)) for fe in self.fes]
        dumb_dofs, self.unflatten_fn_sol_list = jax.flatten_util.ravel_pytree(
            dumb_sol_list
        )
        self.num_total_dofs_all_vars = len(dumb_dofs)

    def _flatten_cells_sol(self, cells_sol_list):
        """Flatten per-cell solution arrays for cell kernels."""
        if self.num_vars == 1:
            return cells_sol_list[0].reshape((self.num_cells, -1))
        return jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(
            *cells_sol_list
        )

    # @timeit
    def split_and_compute_cell(self, cells_sol_flat, np_version, jac_flag, internal_vars):
        """Volume integral in weak form
        """
        vmap_fn = self.kernel_jac if jac_flag else self.kernel
        num_cuts = self._cell_assembly_cut_count()
        input_collection = [cells_sol_flat, self.physical_quad_points, self.shape_grads, self.JxW, self.v_grads_JxW, *internal_vars]
        if num_cuts == 1:
            return vmap_fn(*input_collection)

        batch_size = self.num_cells // num_cuts

        if jac_flag:
            values = []
            jacs = []
            for i in range(num_cuts):
                input_col = _cell_chunk(input_collection, i, batch_size, num_cuts, self.num_cells)

                val, jac = vmap_fn(*input_col)
                values.append(val)
                jacs.append(jac)
            values = np_version.vstack(values)
            jacs = np_version.vstack(jacs)

            return values, jacs
        else:
            values = []
            for i in range(num_cuts):
                input_col = _cell_chunk(input_collection, i, batch_size, num_cuts, self.num_cells)

                val = vmap_fn(*input_col)
                values.append(val)
            values = np_version.vstack(values)
            return values

    def compute_face(self, cells_sol_flat, np_version, jac_flag, internal_vars_surfaces):
        """Surface integral in weak form
        """
        if jac_flag:
            values = []
            jacs = []
            for i, boundary_inds in enumerate(self.boundary_inds_list):
                if boundary_inds.shape[0] == 0:
                    cell_dofs = cells_sol_flat.shape[-1]
                    dtype = getattr(cells_sol_flat, "dtype", None)
                    kwargs = {"dtype": dtype} if dtype is not None else {}
                    values.append(np_version.zeros((0, cell_dofs), **kwargs))
                    jacs.append(np_version.zeros((0, cell_dofs, cell_dofs), **kwargs))
                    continue

                vmap_fn = self.kernel_jac_face[i]
                selected_cell_sols_flat = cells_sol_flat[boundary_inds[:, 0]]  # (num_selected_faces, num_nodes*vec + ...))
                input_collection = [selected_cell_sols_flat, self.physical_surface_quad_points[i], self.selected_face_shape_vals[i],
                                    self.selected_face_shape_grads[i], self.nanson_scale[i], *internal_vars_surfaces[i]]

                val, jac = vmap_fn(*input_collection)
                values.append(val)
                jacs.append(jac)
            return values, jacs
        else:
            values = []
            for i, boundary_inds in enumerate(self.boundary_inds_list):
                if boundary_inds.shape[0] == 0:
                    cell_dofs = cells_sol_flat.shape[-1]
                    dtype = getattr(cells_sol_flat, "dtype", None)
                    kwargs = {"dtype": dtype} if dtype is not None else {}
                    values.append(np_version.zeros((0, cell_dofs), **kwargs))
                    continue

                vmap_fn = self.kernel_face[i]
                selected_cell_sols_flat = cells_sol_flat[boundary_inds[:, 0]]  # (num_selected_faces, num_nodes*vec + ...))
                # TODO: duplicated code
                input_collection = [selected_cell_sols_flat, self.physical_surface_quad_points[i], self.selected_face_shape_vals[i],
                                    self.selected_face_shape_grads[i], self.nanson_scale[i], *internal_vars_surfaces[i]]
                val = vmap_fn(*input_collection)
                values.append(val)
            return values

    def compute_residual_vars_helper(self, weak_form_flat, weak_form_face_flat):
        res_list = [np.zeros((fe.num_total_nodes, fe.vec), dtype=weak_form_flat.dtype) for fe in self.fes]
        if self.num_vars == 1:
            fe = self.fes[0]
            cell_num_nodes = getattr(fe, "num_nodes", self.cells_list[0].shape[1])
            weak_form = weak_form_flat.reshape((-1, cell_num_nodes, fe.vec))
            res_list[0] = res_list[0].at[self.cells_list[0].reshape(-1)].add(
                weak_form.reshape(-1, fe.vec)
            )

            for ind, cells_list_face in enumerate(self.cells_list_face_list):
                if not cells_list_face or cells_list_face[0].shape[0] == 0:
                    continue
                face_cell_num_nodes = cells_list_face[0].shape[1]
                weak_form_face = weak_form_face_flat[ind].reshape(
                    (-1, face_cell_num_nodes, fe.vec)
                )
                res_list[0] = res_list[0].at[
                    cells_list_face[0].reshape(-1)
                ].add(weak_form_face.reshape(-1, fe.vec))

            return res_list

        weak_form_list = jax.vmap(lambda x: self.unflatten_fn_dof(x))(weak_form_flat) # [(num_cells, num_nodes, vec), ...]
        res_list = [res_list[i].at[self.cells_list[i].reshape(-1)].add(weak_form_list[i].reshape(-1,
            self.fes[i].vec)) for i in range(self.num_vars)]

        for ind, cells_list_face in enumerate(self.cells_list_face_list):
            if not cells_list_face or cells_list_face[0].shape[0] == 0:
                continue
            weak_form_face_list = jax.vmap(lambda x: self.unflatten_fn_dof(x))(weak_form_face_flat[ind]) # [(num_selected_faces, num_nodes, vec), ...]
            res_list = [res_list[i].at[cells_list_face[i].reshape(-1)].add(weak_form_face_list[i].reshape(-1,
                self.fes[i].vec)) for i in range(self.num_vars)]

        return res_list

    def compute_residual_vars(self, sol_list, internal_vars, internal_vars_surfaces):
        logger.debug(f"Computing cell residual...")
        cells_sol_list = [sol[cells] for cells, sol in zip(self.cells_list, sol_list)] # [(num_cells, num_nodes, vec), ...]
        cells_sol_flat = self._flatten_cells_sol(cells_sol_list) # (num_cells, num_nodes*vec + ...)
        weak_form_flat = self.split_and_compute_cell(cells_sol_flat, np, False, internal_vars)  # (num_cells, num_nodes*vec + ...)
        weak_form_face_flat = self.compute_face(cells_sol_flat, np, False, internal_vars_surfaces)  # [(num_selected_faces, num_nodes*vec + ...), ...]
        return self.compute_residual_vars_helper(weak_form_flat, weak_form_face_flat)

    def compute_newton_vars(self, sol_list, internal_vars, internal_vars_surfaces):
        logger.debug(f"Computing cell Jacobian and cell residual...")
        cells_sol_list = [sol[cells] for cells, sol in zip(self.cells_list, sol_list)] # [(num_cells, num_nodes, vec), ...]
        cells_sol_flat = self._flatten_cells_sol(cells_sol_list) # (num_cells, num_nodes*vec + ...)
        # (num_cells, num_nodes*vec + ...),  (num_cells, num_nodes*vec + ..., num_nodes*vec + ...)
        if TANGENT_HOST_PATH == "legacy":
            weak_form_flat, cells_jac_flat = self.split_and_compute_cell(cells_sol_flat, onp, True, internal_vars)
            self.V = onp.array(cells_jac_flat.reshape(-1))

            # [(num_selected_faces, num_nodes*vec + ...,), ...], [(num_selected_faces, num_nodes*vec + ..., num_nodes*vec + ...,), ...]
            weak_form_face_flat, cells_jac_face_flat = self.compute_face(cells_sol_flat, onp, True, internal_vars_surfaces)
            for cells_jac_f_flat in cells_jac_face_flat:
                self.V = onp.hstack((self.V, onp.array(cells_jac_f_flat.reshape(-1))))

            return self.compute_residual_vars_helper(weak_form_flat, weak_form_face_flat)

        # cells, then face blocks, straight into the reused host buffer (same order as I/J)
        writer = _TangentHostWriter(self)
        weak_form_flat, cells_jac_flat = self.split_and_compute_cell(cells_sol_flat, writer, True, internal_vars)
        if not writer.wrote_cells:
            writer.write([cells_jac_flat])
        weak_form_face_flat, cells_jac_face_flat = self.compute_face(cells_sol_flat, onp, True, internal_vars_surfaces)
        writer.write(cells_jac_face_flat)
        self.V = writer.finish()
        return self.compute_residual_vars_helper(weak_form_flat, weak_form_face_flat)

    def compute_residual(self, sol_list):
        """Given FE solution list, compute the residual list.

        Parameters
        ----------
        sol_list : list
            A list of JaxArray with the shape being (num_total_nodes, vec).

        Returns
        -------
        res_list : list
            Same shape as sol_list.
        """
        return self.compute_residual_vars(sol_list, self.internal_vars, self.internal_vars_surfaces)

    def newton_update(self, sol_list):
        """Given FE solution list, compute the tangent stiffness matrix, as well as the residual list.

        Parameters
        ----------
        sol_list : list
            A list of JaxArray with the shape being (num_total_nodes, vec).

        Returns
        -------
        res_list : list
            Same shape as sol_list.
            The tangent stiffness matrix is stored internally as instance variables.
        """
        return self.compute_newton_vars(sol_list, self.internal_vars, self.internal_vars_surfaces)

    def set_params(self, params):
        """This is the key method for solving differentiable inverse problems.
        We MUST define (override) this method so that ``params`` become differentiable.
        No need to define this method if only forward problem is solved.

        For parameters defined on the element quadrature points, we may define ::

            def set_params(self, params):
                # Generally, [params1, params2, ...]
                self.internal_vars = [params]

        For parameters defined on the element surface quadrature points, we may define ::

            def set_params(self, params):
                surface_params = params
                # Generally, [[surface1_params1, surface1_params2, ...], [surface2_params1, surface2_params2, ...], ...]
                self.internal_vars_surfaces = [[surface_params]]

        Note that ``params`` itself can be flexible, but ``self.internal_vars`` must accept a precribed input shape.
        The following example inputs ``theta`` as ``params``, and convert it into a well defined ``self.internal_vars``. ::

            def set_params(self, theta):
                # theta is a scalar (float)
                thetas = theta * np.ones((self.fes[0].num_cells, self.fes[0].num_quads))
                # thetas must have the precribed shape to be (num_cells, num_quads, ...)
                self.internal_vars = [thetas]
    
        Then, the following automatic differentiable wrapper must be applied to make ``fwd_pred`` a differentiable function. ::

            fwd_pred = ad_wrapper(problem)
            sol_list = fwd_pred(params)
       

        A similar argument holds for ``self.internal_vars_surfaces`` and ``get_surface_maps``.

        Notes
        -----
        The definition of ``self.internal_vars`` is bonded to ``get_tensor_map``. If ::

            def set_params(self, params):
                # params has shape (num_cells, num_quads, shape1, shape2)
                self.internal_vars = [params]
        
        Then, we must have ::

            class Elasticity(Problem):
                def get_tensor_map(self):
                    def stress_fn(u_grad, param):
                        # param MUST have shape (shape1, shape2)
                        # ...
                        return stress
                    return stress_fn

                def get_mass_map(self):
                    def mass_fn(u, x, param):
                        # param MUST have shape (shape1, shape2)
                        # ...
                        return stress
                    return mass_fn 

        params: `JaxPytree <https://docs.jax.dev/en/latest/pytrees.html>`_
            The parameters to be differentiated. 
        """
        raise NotImplementedError("Child class must implement this function!")

    def print_BC_info(self):
        """Print boundary and surface-integral set information for debugging."""
        boundary_inds_list = self.boundary_inds_list
        if len(boundary_inds_list) != 0:
            print("\n\n### Surface integral boundary sets are specified")
            for i in range(len(boundary_inds_list)):
                print(f"\nSurface boundary set {i + 1} information:")
                print(boundary_inds_list[i])
                print(
                    f"Array.shape = (num_selected_faces, 2) = {boundary_inds_list[i].shape}"
                )
                print("Interpretation:")
                print(
                    "    Array[i, 0] returns the global cell index of the ith selected face"
                )
                print(
                    "    Array[i, 1] returns the local face index of the ith selected face"
                )
        else:
            print("\n\n### No surface integral boundary sets found.")

        for var_i, fe in enumerate(self.fes):
            var_suffix = (
                f" (finite element variable {var_i})"
                if self.num_vars > 1
                else ""
            )
            if len(fe.node_inds_list) != 0:
                print(f"\n\n### Dirichlet B.C. is specified{var_suffix}")
                for i in range(len(fe.node_inds_list)):
                    print(f"\nDirichlet Boundary part {i + 1} information:")
                    if len(fe.node_inds_list[i]) == 0:
                        bc_array = onp.zeros((0, 3))
                    else:
                        bc_array = onp.stack([
                            fe.node_inds_list[i],
                            fe.vec_inds_list[i],
                            fe.vals_list[i],
                        ]).T
                    print(bc_array)
                    print(
                        f"Array.shape = (num_selected_dofs, 3) = {bc_array.shape}"
                    )
                    print("Interpretation:")
                    print(
                        "    Array[i, 0] returns the node index of the ith selected dof"
                    )
                    print(
                        "    Array[i, 1] returns the vec index of the ith selected dof"
                    )
                    print(
                        "    Array[i, 2] returns the value assigned to ith selected dof"
                    )
            else:
                print(f"\n\n### No Dirichlet B.C. found{var_suffix}.")
