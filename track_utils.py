import numpy as np
from scipy.ndimage import center_of_mass

def sample_points_from_masks(masks, num_points):
    """
    sample points from masks and return its absolute coordinates

    Args:
        masks: np.array with shape (n, h, w)
        num_points: int

    Returns:
        points: np.array with shape (n, points, 2)
    """
    n, h, w = masks.shape
    points = []

    for i in range(n):
        # 找到当前mask中值为1的位置的坐标
        indices = np.argwhere(masks[i] == 1)  
        # the output format of np.argwhere is (y, x) and the shape is (num_points, 2)
        # we should convert it to (x, y)
        indices = indices[:, ::-1]  # (num_points, [y x]) to (num_points, [x y])
        
        # import pdb; pdb.set_trace()
        if len(indices) == 0:
            # 如果没有有效点，返回一个空数组
            points.append(np.array([]))
            continue
        
        # 如果mask中的点少于需要的数量，则重复采样
        if len(indices) < num_points:
            sampled_indices = np.random.choice(len(indices), num_points, replace=True)
        else:
            sampled_indices = np.random.choice(len(indices), num_points, replace=False)
        
        sampled_points = indices[sampled_indices]
        points.append(sampled_points)

    # 将结果转换为numpy数组
    points = np.array(points, dtype=np.float32)
    return points
