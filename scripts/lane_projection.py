import os
import numpy as np
import array
import cv2


def load_pc(bin_file_path):
    """
    load pointcloud file (velodyne format)
    :param bin_file_path:
    :return:
    """
    with open(bin_file_path, 'rb') as bin_file:
        pc = array.array('f')
        pc.frombytes(bin_file.read())
        pc = np.array(pc).reshape(-1, 4)
        return pc

def load_calib2(calib_file_path):
    """
    load calibration file(KITTI object format)
    :param calib_file_path:
    :return:
    """
    calib_file = open(calib_file_path, 'r').readlines()
    calib_file = [line
                      .replace('Tr_velo_to_cam', 'Tr_velo_cam')
                      .replace('R0_rect', 'R_rect')
                      .replace('\n', '')
                      .replace(':', '')
                      .split(' ')
                  for line in calib_file]
    calib_file = {line[0]: [float(item) for item in line[1:] if item != ''] for line in calib_file if len(line) > 1}
    return calib_file

def load_calib(calib_path):
    f = open(calib_path, "r")
    line = f.readline()
    calib = {}
    while line:
        tmp = line.split(":")
        tr = tmp[1].split(" ")[1:]
        tr = list(map(eval, tr))
        m = np.identity(4)
        for j in range(12):
            m[j//4, j- (j//4)*4] = tr[j]
        calib[tmp[0]] = np.mat(m)
        line = f.readline()
    f.close()
    return calib


def load_pose(pose_path):
    f = open(pose_path,"r")
    line = f.readline()[:-1]
    poses = []
    while line:
        poses.append(np.array([float(m) for m in line.split(' ')]))
        line = f.readline()[:-1]
    f.close()
    res = []
    for i in range(len(poses)):
        pose = np.identity(4)
        for j in range(12):
            pose[j//4, j - (j//4)*4] = poses[i][j]
        res.append(pose)
    return res



def parse_calib_file(calib_file):
    """
    parse calibration file to calibration matrix
    :param calib_file:
    :return:
    """

    # 外参矩阵
    Tr_velo_cam = np.array(calib_file['Tr_velo_cam']).reshape(3, 4)
    Tr_velo_cam = np.concatenate([Tr_velo_cam, [[0, 0, 0, 1]]], axis=0)
    # 矫正矩阵
    R_rect = np.array(calib_file['R_rect']).reshape(3, 3)
    R_rect = np.pad(R_rect, [[0, 1], [0, 1]], mode='constant')
    R_rect[-1, -1] = 1
    # 内参矩阵
    P2 = np.array(calib_file['P2']).reshape(3, 4)

    return np.matmul(np.matmul(P2, R_rect), Tr_velo_cam)
  
def get_project2image_lane_points(pc, img_size, calib_file, label, yaw_deg):
    """
    获取点云的前视投影
    :param pc: 输入点云(N, 4)
    :param img_size: (w, h)
    :param calib_file: KITTI calib文件的path
    :param yaw_deg: 将点云按z轴旋转的角度，默认设置为0就可以
    :return:
    """
    yaw_deg = yaw_deg / 180 * np.pi
    calib_mat = parse_calib_file(calib_file)
    # calib_mat = parse_calib_file(load_calib(calib_file))

    # 投影
    intensity = np.copy(pc[:, 3]).reshape(-1, 1)
    height = np.copy(pc[:, 2]).reshape(-1, 1)
    pc[:, 3] = 1
    # yaw旋转
    rotate_mat = np.array([
        [np.cos(yaw_deg), -np.sin(yaw_deg), 0, 0],
        [np.sin(yaw_deg), np.cos(yaw_deg), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    pc = np.matmul(rotate_mat, pc.T).T

    # 限制前视 并且限制fov在90度之内
    # 计算fov
    fov_h = np.arctan(np.abs(pc[:, 1] / pc[:, 0]))
    fov_v = np.arctan(np.abs(pc[:, 2] / pc[:, 0]))
    indice = np.where(np.bitwise_and(
        pc[:, 0] > 0.5,
        np.bitwise_and(fov_h < np.pi / 4, fov_v < np.pi / 10, )
    ))
    pc = pc[indice]
    print(pc.shape)

    # mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color=(1, 0, 0), mode='point')
    # pc = pc[np.random.permutation(len(pc))[:28000], :]
    # mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color=(0, 1, 0), mode='point')
    # mlab.show()

    intensity = intensity[indice]
    height = height[indice]
    label = label[indice]
    label=label[:, np.newaxis]

    # 进行投影变换
    pc = np.matmul(calib_mat, pc.T).T
    
    # z深度归一化
    pc[:, :2] /= pc[:, 2:]
    
    # 还原intensity
    pc = np.concatenate([pc, intensity, height, label], axis=1)
    
    # 按照原图大小裁剪
    pc = pc[np.where(pc[:, 0] >= 0)]
    pc = pc[np.where(pc[:, 0] < img_size[0])]
    pc = pc[np.where(pc[:, 1] >= 0)]
    pc = pc[np.where(pc[:, 1] < img_size[1])]

    lane_pc = np.where(pc[:, 5]==1)
    lane_point = pc[lane_pc]
    return np.int_(lane_point[:, 1]), np.int_(lane_point[:, 0])

def get_superposition_image(basic_pth, index):
    calib_path = os.path.join(basic_path, 'calib.txt')
    pose_path = os.path.join(basic_path, 'poses.txt')
    pc_array = []
    poses = load_pose(pose_path)
    indexs = np.arange(index, index+50)
    calib = load_calib(calib_path)
    for i in range(len(poses)):
        Tr = calib['Tr']
        Tr = np.mat(Tr)
        Tr_inv = Tr.I
        poses[i] = np.dot(np.dot(Tr_inv, poses[i]), Tr)
    T_left = poses[index].I
    res = []
    colors = []
    remissions = []
    labels = []
    frame_index = index
    for index in indexs:
        bin_path = os.path.join(basic_path,'velodyne', str(index).zfill(6)+'.bin')
        label_path = os.path.join(basic_path, 'labels', str(index).zfill(6)+'.label')
        label = np.fromfile(label_path, dtype=np.uint32)
        labels.append(label)
        pc = load_pc(bin_path)
        remission = pc[:,3].copy()
        remissions.append(remission)
        # pc[:,3] = 1
        pc[:, 3] = 1
        pc = np.transpose(pc)
        pc_squence = np.dot(T_left,np.dot(poses[index], pc))
        res.append(np.transpose(pc_squence.A))
        tmp = np.array(np.ceil(remission*255))
    labels = np.concatenate(labels)
    label = labels
    pc_array = np.array(res)
    pc_array = np.concatenate(pc_array)
    pc_array[:,3] = np.concatenate(remissions)
    index = str(frame_index).zfill(6)
    calib_path = os.path.join(basic_path, 'calib2.txt')
    calib = load_calib2(calib_path)
    img_path = os.path.join(basic_path, 'image_2', index+'.png')
    img = cv2.imread(img_path)
    cv2.imwrite('fov.png', img)
    img_size = img.T.shape[1:]
    xs,ys = get_project2image_lane_points(pc_array, img_size, calib, label, yaw_deg=0)
    for i in range(len(xs)):
        cv2.circle(img, (ys[i],xs[i]), 1, [0,0,255], 4)
    # cv2.imshow('', img)
    # cv2.waitKey(0)
    cv2.imwrite('fov_lane.png', img)


if __name__ == '__main__':
    index = 50
    basic_path = '/home/fzkgod/dataset/dataset/kittivo/sequences/01'
    get_superposition_image(basic_path, index)
    
    