"""
Implements the complete synthesis pipeline by tying together methods from all the other fiels.
"""
import torch
import articulate as art


def transform_poses_trans(poses, trans):
    """
    We'll discard the hand part of the poses and align the AMASS global frame
    with that of DIP by applying rotations. Furthermore, we'll group together
    the angles from the same joints, adding a dimension with size 3 to the
    shape.
    """

    poses = poses.view(-1, 52, 3) # 52 joints total, 3 angles per joint
    poses[:, 23] = poses[:, 37] # extract the right hand
    poses = poses[:, :24].clone() # discard the articulated hand

    # align AMASS global frame with that of with DIP. Essentially, this means
    # that we're going to rotate the pelvis of all the AMASS data.
    amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
    trans = amass_rot.matmul(trans.unsqueeze(-1)).view_as(trans)
    poses[:, 0] = art.math.rotation_matrix_to_axis_angle(
        amass_rot.matmul(
            art.math.axis_angle_to_rotation_matrix(poses[:, 0])
        )
    )

    poses = art.math.axis_angle_to_rotation_matrix(poses).view(-1, 24, 3, 3)

    return poses, trans

def run_kinematics(model, poses, trans, betas):
    """
    We'll run kinematics on the pose data here to obtain rotations, joints, and vertices.
    The model passed in should be raw SMPL, not SMPL+H.

    Returns rotations, joints, and vertices
    
    """
    grot, joint, vert = model.forward_kinematics(
        poses,
        betas,
        trans,
        calc_mesh=True
    ) 

    return grot, joint[:, :24].contiguous().clone(), vert

def normalize_imu_data(acc, ori, num_joints):
    glb_acc = acc.view(-1, num_joints, 3)
    glb_ori = ori.view(-1, num_joints, 3, 3)
    
    root_acc = glb_acc[:, :1]
    other_acc = glb_acc[:, 1:]

    root_ori = glb_ori[:, :1]
    other_ori = glb_ori[:, 1:]

    acc = torch.cat((root_acc, other_acc - root_acc), dim=1).bmm(glb_ori[:, 0]) / 30
    ori = torch.cat((root_ori, root_ori.transpose(2, 3).matmul(other_ori)), dim=1)

    return acc, ori

def synthesize_imu_data(vert_model: art.ParametricModel, vert, grot, requested_joints, smooth_n=4):
    """
    We'll synthesize accelerations by extracting the vertices that correspond to
    the requested joints on the SMPL model, passed in as vert_model.

    We'll also grab just the rotations corresponding to the requested joints.

    Returns acc, rot
    """
    vertices = vert_model._J_regressor.argmax(axis=1)[requested_joints]
    v = vert[:, vertices]

    # synthesize accelerations
    mid = smooth_n // 2
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
                for i in range(0, v.shape[0] - smooth_n * 2)])

    rot = grot[:, requested_joints]   

    norm_acc, norm_rot = normalize_imu_data(acc, rot, len(requested_joints))

    return norm_acc, norm_rot

def generate_synthesized_sample(model, sequence, requested_joints, smooth_n=4):
    """
    Runs the entire preprocess pipeline on a single sequence. 
    """
    poses, trans, betas = sequence
    poses, trans = transform_poses_trans(poses, trans)
    grot, joint, vert = run_kinematics(model, poses, trans, betas)
    imu_acc, imu_rot = synthesize_imu_data(model, vert, grot, requested_joints, smooth_n)

    return poses, trans, betas, joint, imu_acc, imu_rot