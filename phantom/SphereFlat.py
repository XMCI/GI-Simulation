import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
sys.path.append('.//configs//')

def relaxation_pack(num_circles, radius, domain_size, 
                    max_iter=5000, learning_rate=0.1, tolerance=1e-4):
    """
    使用改进的力导向松弛算法打包更多圆，并记录迭代过程中的“能量”（overlap 能量）以示下降。
    
    参数:
    - num_circles: 圆的数量
    - radius: 圆的半径
    - domain_size: 正方形区域边长
    - max_iter: 最大迭代次数
    - learning_rate: 更新步长（梯度下降的学习率）
    - tolerance: 当能量低于此阈值时提前终止
    
    返回:
    - centers: 最终圆心数组，形状为 (num_circles, 2)
    - overlap_history: 长度不超过 max_iter 的列表，记录每次迭代的 max_overlap 
    """
    # 初始化圆心在 [radius, domain_size - radius] 区域内
    centers = np.random.uniform(radius, domain_size - radius, size=(num_circles, 2))
    
    overlap_history = []
    
    for it in range(max_iter):
        # 计算所有圆心对的向量差和距离
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]  # shape (N, N, 2)
        distances = np.linalg.norm(diff, axis=2)  # shape (N, N)
        
        # 只关心 i < j 的一半矩阵，避免双重计算
        i_idx, j_idx = np.triu_indices(num_circles, k=1)
        pair_dists = distances[i_idx, j_idx]
        
        # 计算 overlap：如果 d < 2R，则 overlap = 2R - d，否则为 0
        overlaps = np.maximum(0.0, 2 * radius - pair_dists)
        
        # 计算最大重叠
        max_overlap = np.max(overlaps)
        overlap_history.append(max_overlap)
        
        # 如果能量很小，提前结束
        if max_overlap < tolerance:
            print(f"迭代在第 {it} 步收敛，最大重叠 < {tolerance}")
            break
        if (it % 5000 == 0 or it == max_iter - 1):
            print(f"Iter {it:4d}, max_overlap = {max_overlap:.6f}")
        
        # 初始化合力矩阵
        forces = np.zeros_like(centers)  # shape (N, 2)
        
        # 对于每个重叠对 (i, j)，计算分离力
        # 力的大小与 overlap 成正比，方向为单位向量 diff / d
        # 施加到 i 上方向为 +dir，施加到 j 上方向为 -dir
        for idx in range(len(i_idx)):
            i = i_idx[idx]
            j = j_idx[idx]
            d = pair_dists[idx]
            ov = overlaps[idx]
            if ov > 0:
                # 若两个圆心完全重合，随机给一个方向
                if d == 0:
                    direction = np.random.uniform(-1, 1, size=2)
                    direction /= np.linalg.norm(direction)
                else:
                    direction = diff[i, j] / d
                # 力的大小可取 ov 本身
                f = ov * direction
                forces[i] += f
                forces[j] -= f
        
        # 更新圆心位置：梯度下降一步
        centers += learning_rate * forces
        
        # 保持边界约束：圆心在 [radius, domain_size-radius]
        centers = np.clip(centers, radius, domain_size - radius)

    return centers, overlap_history

def compute_thickness_map(centers, radius, domain_size, grid_resolution):
    """
    给定圆心列表和半径，在一个 grid_resolution 大小的网格上计算投影厚度（球面高度）。
    对于每个圆，若网格点 (x, y) 落在圆内，则厚度 h = sqrt(radius^2 - ((x-cx)^2 + (y-cy)^2))，
    否则厚度为 0。由于圆<->球投影互不重叠，可直接赋值。

    参数:
    - centers: 形状为 (num_centers, 2) 的圆心数组
    - radius: 圆/球的半径
    - domain_size: 方形区域边长
    - grid_resolution: (ny, nx)，生成的厚度图像大小

    返回:
    - thickness_map: 形状等同 grid_resolution 的二维数组，存储每个像素的厚度
    """
    ny, nx = grid_resolution
    x = np.linspace(0, domain_size, nx)
    y = np.linspace(0, domain_size, ny)
    xv, yv = np.meshgrid(x, y)

    thickness_map = np.zeros_like(xv)

    for cx, cy in centers:
        dx = xv - cx
        dy = yv - cy
        distances = np.sqrt(dx**2 + dy**2)
        within_circle = distances <= radius
        height = np.sqrt(radius**2 - distances**2)

        # 圆之间不重叠，所以直接赋值
        thickness_map[within_circle] = height[within_circle]

    return thickness_map*2

# —— 使用示例 —— 
if __name__ == "__main__":
    domain_size = 2.4e-3    # 区域边长 unit: m 就是模拟中的FOV
    radius = 100e-6           # 圆/球半径 unit: m
    slice_thickness = 2 * radius  # 层厚 unit: m
    n_slice = 25          # 层数
    V = 4/3 * np.pi * radius**3  # 球体积
    VF = 0.4          # 体积分数
    num_circles = int((VF * domain_size**2 * slice_thickness) / V)  # 计算所需的圆的数量
    gap = 0  # 层间距 unit: m
    save_dir = 'phantom//'
    
    # 松弛算法参数
    max_iter = 30000
    learning_rate = 0.05  # 适当调小学习率，避免震荡
    tolerance = 1e-7

    grid_resolution = (4000, 4000)  # 输出厚度图尺寸
    Sph = np.zeros((n_slice, grid_resolution[0], grid_resolution[1]))  # 初始化厚度图
    for i in range(n_slice):
        # 1. 生成互不重叠圆心
        centers, energy_hist = relaxation_pack(num_circles, radius, domain_size, max_iter, learning_rate, tolerance)
        # print(f"最终圆心分布: {centers[0]}")

        # 2. 计算投影厚度图
        thickness_map = compute_thickness_map(centers, radius, domain_size, grid_resolution)

        Sph[i, :, :] = thickness_map

        print(f"第 {i+1} 层的厚度图计算完成")

    np.savez(save_dir + 'Sph_' + '200' + 'um_'+str(VF)+'_2D.npz', 
            Sph=Sph, 
            gap=gap, 
            slice_thickness=slice_thickness,
            N_slice = n_slice)
    



    # # 3. 可视化结果
    # plt.figure(figsize=(8, 8))
    # plt.imshow(thickness_map, cmap='gray', extent=(0, domain_size, 0, domain_size))
    # plt.colorbar(label='projection thickness (m)')
    # plt.title(f"(num_circles={num_circles}, radius={radius})")
    # plt.xlabel("x (m)")
    # plt.ylabel("y (m)")

    # plt.figure(figsize=(8, 8))
    # plt.plot(thickness_map[int(centers[0,0]),:])

    # plt.show()

    
    
    # # 绘制能量下降曲线
    # plt.figure()
    # plt.plot(energy_hist)
    # plt.xlabel("迭代次数")
    # plt.ylabel("总 overlap 能量")
    # plt.title("迭代过程中的能量下降")
    # plt.show()

    # # 绘制最终圆心分布示意
    # plt.figure()
    # plt.scatter(centers[:, 0], centers[:, 1], s=20)
    # circle = plt.Circle((0, 0), radius=radius, color='black', fill=False)
    # plt.gca().add_patch(plt.Rectangle((0, 0), domain_size, domain_size,
    #                                 edgecolor='black', fill=False))
    # # 单独画一个圆示范
    # for (cx, cy) in centers:
    #     c = plt.Circle((cx, cy), radius=radius, edgecolor='blue', facecolor='none', alpha=0.7)
    #     plt.gca().add_patch(c)
    # plt.xlim(0, domain_size)
    # plt.ylim(0, domain_size)
    # plt.gca().set_aspect('equal', 'box')
    # plt.title("最终圆心分布（蓝色圈示意）")
    # plt.show()
