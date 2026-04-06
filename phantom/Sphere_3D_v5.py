import numpy as np
from multiprocessing import Pool, cpu_count
from multiprocessing import shared_memory
import os
from tqdm import tqdm
from scipy.ndimage import zoom
import math

# Global worker variables (set by initializer)
_centers_shm = None
_centers_shape = None
_centers_dtype = None
_centers = None
_worker_rng = None


def _init_worker(shm_name, shape, dtype, rng_seed_offset=0):
    """
    Pool initializer: connect each subprocess to the shared memory block 
    and create a worker-local RNG instance.
    """
    global _centers_shm, _centers_shape, _centers_dtype, _centers, _worker_rng
    _centers_shape = tuple(shape)
    _centers_dtype = np.dtype(dtype)
    _centers_shm = shared_memory.SharedMemory(name=shm_name)
    _centers = np.ndarray(_centers_shape, dtype=_centers_dtype, buffer=_centers_shm.buf)

    # Each worker uses a different seed based on its PID
    seed = (os.getpid() + rng_seed_offset) & 0xFFFFFFFF
    _worker_rng = np.random.default_rng(seed)


def process_chunk(args):
    """
    Worker-side computation:
      - Read the shared global `_centers`
      - Compute displacements for a given pair index subset (i_sub, j_sub)
    
    Returns:
      i_sub_valid, j_sub_valid : indices for overlapping sphere pairs
      disp_i, disp_j           : displacement vectors
      chunk_max_overlap        : maximum overlap in this chunk
    """
    global _centers, _worker_rng
    i_sub, j_sub, R = args
    if _centers is None:
        raise RuntimeError("Worker has no access to shared centers array")

    # Vector differences between sphere centers
    diffs = _centers[i_sub] - _centers[j_sub]
    dists = np.linalg.norm(diffs, axis=1)
    valid_mask = dists < 2 * R

    # If no overlap, return empty arrays
    if not np.any(valid_mask):
        return (np.array([], dtype=int), np.array([], dtype=int),
                np.zeros((0, 3)), np.zeros((0, 3)), 0.0)

    valid_diffs = diffs[valid_mask]
    valid_dists = dists[valid_mask]
    i_sub_valid = i_sub[valid_mask]
    j_sub_valid = j_sub[valid_mask]

    # Avoid division by zero for nearly coincident spheres
    eps = 1e-12
    small_mask = valid_dists < eps
    valid_dists_safe = valid_dists.copy()
    valid_dists_safe[small_mask] = 1.0  # placeholder, replaced later

    dir_ij = valid_diffs / valid_dists_safe[:, np.newaxis]

    # Replace directions for nearly coincident points with random unit vectors
    if np.any(small_mask):
        n_small = np.sum(small_mask)
        rnd = _worker_rng.normal(size=(n_small, 3))
        rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
        dir_ij[small_mask] = rnd

    overlaps = 2 * R - valid_dists
    chunk_max_overlap = overlaps.max() if overlaps.size > 0 else 0.0

    disp_i = 0.5 * overlaps[:, np.newaxis] * dir_ij
    disp_j = -disp_i

    return (i_sub_valid, j_sub_valid, disp_i, disp_j, float(chunk_max_overlap))


def relax_sphere_centers(N, R, L_vec, max_iter=1000, dt=0.1, tol=1e-3, verbose=True, num_cores=None):
    """
    Optimized sphere-relaxation algorithm:
      - Uses shared memory to avoid pickling sphere centers
      - Pool is created once outside the iteration loop and reused
      - Main process merges duplicated indices before applying displacements

    Parameters:
      N           number of spheres
      R           sphere radius
      L_vec       simulation box size [Lx, Ly, Lz]
      max_iter    maximum iterations
      dt          relaxation step size
      tol         convergence tolerance
    """
    if num_cores is None:
        num_cores = cpu_count()

    np.random.seed(None)
    L_vec = np.asarray(L_vec, dtype=float).reshape(3)
    low = np.array([R, R, R], dtype=float)
    high = L_vec - R

    # Random initial sphere locations
    centers = np.random.uniform(low=low, high=high, size=(N, 3)).astype(np.float64)

    # Create shared memory and initialize with centers
    shm = shared_memory.SharedMemory(create=True, size=centers.nbytes)
    try:
        shm_buf = np.ndarray(centers.shape, dtype=centers.dtype, buffer=shm.buf)
        np.copyto(shm_buf, centers)

        # Precompute the index pairs (upper triangle only)
        i_list_full, j_list_full = np.triu_indices(N, k=1)
        if i_list_full.size == 0:
            if verbose:
                print("No sphere pairs to process (N <= 1). Returning initial centers.")
            return centers

        chunk_size = max(1, len(i_list_full) // (num_cores * 8))
        chunks = []
        for k in range(0, len(i_list_full), chunk_size):
            chunks.append((i_list_full[k: k + chunk_size],
                           j_list_full[k: k + chunk_size], R))

        # Create worker pool and attach to shared memory
        with Pool(processes=num_cores,
                  initializer=_init_worker,
                  initargs=(shm.name, centers.shape, centers.dtype.str)) as pool:

            if verbose:
                print(f"Using pool with {num_cores} workers, {len(chunks)} chunks per iteration")

            for it in range(max_iter):
                displacements = np.zeros_like(centers)
                max_overlap = 0.0

                # Update shared memory with latest centers
                np.copyto(shm_buf, centers)

                # Parallel processing
                results = pool.map(process_chunk, chunks)

                # Merge displacements
                for i_sub, j_sub, disp_i, disp_j, chunk_max in results:
                    if i_sub.size == 0:
                        continue

                    # Merge duplicate indices for i_sub
                    if i_sub.size > 0:
                        uniq_i, inv_i = np.unique(i_sub, return_inverse=True)
                        summed_i = np.zeros((uniq_i.size, 3), dtype=displacements.dtype)
                        np.add.at(summed_i, inv_i, disp_i)
                        displacements[uniq_i] += summed_i

                    # Merge duplicate indices for j_sub
                    if j_sub.size > 0:
                        uniq_j, inv_j = np.unique(j_sub, return_inverse=True)
                        summed_j = np.zeros((uniq_j.size, 3), dtype=displacements.dtype)
                        np.add.at(summed_j, inv_j, disp_j)
                        displacements[uniq_j] += summed_j

                    if chunk_max > max_overlap:
                        max_overlap = chunk_max

                centers += dt * displacements
                centers = np.clip(centers, low, high)

                if verbose and (it % 50 == 0 or it == max_iter - 1):
                    print(f"Iter {it:4d}, max_overlap = {max_overlap:.8e}")

                if max_overlap < tol:
                    if verbose:
                        print(f"Converged at iter {it}, max_overlap < {tol}")
                    break

        return centers

    finally:
        # Free shared memory
        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass


# --------------------------- Rasterization section ---------------------------

def _rasterize_single_sphere(args):
    """
    Rasterize one sphere to a voxel block.

    Returns the bounding index range and a 3D mask block.
    """
    center, R, L_vec, coords_x, coords_y, coords_z = args
    cx, cy, cz = center

    # Compute bounding box in physical coordinates
    x_min = cx - R
    x_max = cx + R
    y_min = cy - R
    y_max = cy + R
    z_min = cz - R
    z_max = cz + R

    # Convert to voxel index range
    ix_start = np.searchsorted(coords_x, x_min, side='left')
    ix_end   = np.searchsorted(coords_x, x_max, side='right')
    iy_start = np.searchsorted(coords_y, y_min, side='left')
    iy_end   = np.searchsorted(coords_y, y_max, side='right')
    iz_start = np.searchsorted(coords_z, z_min, side='left')
    iz_end   = np.searchsorted(coords_z, z_max, side='right')

    # If outside the grid, return empty
    if ix_start >= ix_end or iy_start >= iy_end or iz_start >= iz_end:
        return ix_start, ix_end, iy_start, iy_end, iz_start, iz_end, np.zeros((0,0,0), dtype=np.uint8)

    xs = coords_x[ix_start:ix_end]
    ys = coords_y[iy_start:iy_end]
    zs = coords_z[iz_start:iz_end]

    xv, yv, zv = np.meshgrid(xs, ys, zs, indexing='ij')
    dx = xv - cx
    dy = yv - cy
    dz = zv - cz
    dist2 = dx*dx + dy*dy + dz*dz
    block_mask = (dist2 <= R*R).astype(np.uint8)

    return ix_start, ix_end, iy_start, iy_end, iz_start, iz_end, block_mask


def rasterize_spheres_to_grid(centers, R, L_vec, grid_size_max, use_progress=True, num_cores=None):
    """
    Rasterize all spheres into a uniform 3D voxel grid.
    """
    L_vec = np.asarray(L_vec, dtype=float).reshape(3)
    max_L = np.max(L_vec)

    # Compute grid resolution proportional to box size
    Gx = max(int(round(grid_size_max * (L_vec[0] / max_L))), 1)
    Gy = max(int(round(grid_size_max * (L_vec[1] / max_L))), 1)
    Gz = max(int(round(grid_size_max * (L_vec[2] / max_L))), 1)

    # Uniform cell centers
    coords_x = np.linspace(0, L_vec[0], Gx, endpoint=False) + (L_vec[0] / Gx) / 2
    coords_y = np.linspace(0, L_vec[1], Gy, endpoint=False) + (L_vec[1] / Gy) / 2
    coords_z = np.linspace(0, L_vec[2], Gz, endpoint=False) + (L_vec[2] / Gz) / 2

    voxels = np.zeros((Gx, Gy, Gz), dtype=np.uint8)

    N = centers.shape[0]
    tasks = [(centers[k], R, L_vec, coords_x, coords_y, coords_z) for k in range(N)]

    if num_cores is None:
        num_cores = cpu_count()

    with Pool(processes=num_cores) as pool:
        iterator = pool.imap(_rasterize_single_sphere, tasks)
        if use_progress:
            iterator = tqdm(iterator, total=N, desc="Rasterizing spheres")

        for ix_start, ix_end, iy_start, iy_end, iz_start, iz_end, block_mask in iterator:
            if block_mask.size == 0:
                continue
            voxels[ix_start:ix_end, iy_start:iy_end, iz_start:iz_end] |= block_mask

    return voxels


if __name__ == "__main__":

    # ====== Parameter settings ======
    total_FOV = 2.4e-3
    total_nP = 4000
    det_pixelSize = 6e-4
    dx = 6e-7

    FOV_center = total_FOV - 2 * det_pixelSize
    nP_center = int(round(FOV_center / dx))

    Lx, Ly, Lz = FOV_center, FOV_center, FOV_center
    L_vec = np.array([Lx, Ly, Lz], dtype=float)

    diameter = [150e-6]

    max_iter = 3000
    dt = 0.2
    tol = 1e-8

    group = 1
    n_slice = 40
    N_pixel = 50
    scale_factor = 1
    slice_thickness = N_pixel * dx * scale_factor
    gap = 0
    save_dir = 'phantom'
    os.makedirs(save_dir, exist_ok=True)

    for D in diameter:
        R = D / 2
        V_sphere = 4 * np.pi * R**3 / 3
        VF = 0.55
        total_volume = Lx * Ly * Lz
        N = int(VF * total_volume / V_sphere)
        print(f"Volume fraction VF={VF}; Number of spheres N = {N}")

        save_name = f"Sph_{int(D*1e6)}um_{VF}.npz"

        Sph = np.zeros((n_slice * group, total_nP, total_nP))

        start_row = int((Sph.shape[1] - nP_center) // 2)
        start_col = int((Sph.shape[2] - nP_center) // 2)
        end_row = int(start_row + nP_center)
        end_col = int(start_col + nP_center)

        for i in range(group):
            print(f"Group {i+1}/{group}: Generating sphere centers...")
            centers = relax_sphere_centers(N=N, R=R, L_vec=L_vec,
                                           max_iter=max_iter, dt=dt,
                                           tol=tol, verbose=True)

            print("Rasterizing...")
            voxels = rasterize_spheres_to_grid(centers, R, L_vec, nP_center)

            total_voxels = voxels.size
            num_inside = int(np.sum(voxels))
            print(f"Total voxels: {total_voxels} (grid shape {voxels.shape}), "
                  f"voxels = 1 count = {num_inside} ({num_inside/total_voxels:.3%})")

            for j in tqdm(range(n_slice), desc=f"Slicing group {i+1}"):
                temp = np.sum(voxels[j*N_pixel:(j+1)*N_pixel, :, :], axis=0)
                Sph[i*n_slice+j, start_row:end_row, start_col:end_col] = zoom(temp, scale_factor, order=3)

            print(f"Finished group {i+1}/{group}")

        np.savez(os.path.join(save_dir, save_name),
                Sph=Sph * scale_factor * dx,
                gap=gap,
                slice_thickness=slice_thickness,
                total_thickness=slice_thickness * n_slice * group,
                N_slice=n_slice)

        print(f"Saved data to {os.path.join(save_dir, save_name)}")
