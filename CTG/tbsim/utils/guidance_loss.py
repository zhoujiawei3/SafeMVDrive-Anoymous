import time

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import tbsim.utils.geometry_utils as GeoUtils
from tbsim.utils.metrics import batch_detect_off_road

from tbsim.utils.geometry_utils import (
    transform_agents_to_world,
    detect_collision
)
from tbsim.utils.trajdata_utils import get_current_lane_projection, get_left_lane_projection, get_right_lane_projection, select_agent_ind, transform_coord_agents_to_world, transform_coord_world_to_agent_i

from torch.autograd import Variable
import tbsim.utils.tensor_utils as TensorUtils
from shapely.geometry import LineString, Polygon
### utils for choosing from samples ####
from typing import Tuple
def choose_action_from_guidance(preds, obs_dict, guide_configs, guide_losses):
    '''
    preds: dict of predictions from model, preds["positions"] (M, N, T, 2) or (B, N, M, T, 2)
    '''
    if len(preds["positions"].shape) == 4:
        # for agent-centric model the batch dimension is always 1
        B = 1
        M, N, *_ = preds["positions"].shape
    else:
        B, N, M, *_ = preds["positions"].shape
    BM = B*M
    # arbitrarily use the first sample as the action if no guidance given
    act_idx = torch.zeros((BM), dtype=torch.long, device=preds["positions"].device)
    # choose sample closest to desired guidance
    accum_guide_loss = torch.stack([v for k,v in guide_losses.items()], dim=2)
    # each scene separately since may contain different guidance
    scount = 0
    for sidx in range(len(guide_configs)):
        scene_guide_cfg = guide_configs[sidx]
        ends = scount + len(scene_guide_cfg)
        # (BM, N, num_of_guidance)
        scene_guide_loss = accum_guide_loss[..., scount:ends]
        scount = ends
        # scene_mask = ~torch.isnan(torch.sum(scene_guide_loss, dim=[1,2]))
        # scene_guide_loss = scene_guide_loss[scene_mask].cpu()
        # (BM, N, num_of_guidance) -> (BM, N)
        scene_guide_loss = torch.nansum(scene_guide_loss, dim=-1)
        is_scene_level = np.array([guide_cfg.name in ['agent_collision', 'social_group', 'gptcollision', 'gptkeepdistance'] for guide_cfg in scene_guide_cfg])
        if np.sum(is_scene_level) > 0: 
            # choose which sample minimizes at the scene level (where each sample is a "scene")
            # (1)
            scene_act_idx = torch.argmin(torch.sum(scene_guide_loss, dim=0))
            # (BM,N) -> (B,M,N) -> (B,N) -> (B)
            scene_act_idx = torch.argmin(scene_guide_loss.reshape(B, M, N).sum(dim=1), dim=1)
            scene_act_idx = scene_act_idx.unsqueeze(-1).expand(B,M).view(BM)
        else:
            # each agent can choose the sample that minimizes guidance loss independently
            # (BM)
            scene_act_idx = torch.argmin(scene_guide_loss, dim=-1)

        # act_idx[scene_mask] = scene_act_idx.to(act_idx.device)
        act_idx = scene_act_idx.to(act_idx.device)

    return act_idx

def choose_action_from_gt(preds, obs_dict):
    '''
    preds: dict of predictions from model, preds["positions"] (M, N, T, 2) or (B, N, M, T, 2)
    '''
    if len(preds["positions"].shape) == 4:
        # for agent-centric model the batch dimension is always 1
        B = 1
        M, N, *_ = preds["positions"].shape
    else:
        B, N, M, *_ = preds["positions"].shape
    BM = B*M

    # arbitrarily use the first sample as the action if no gt given
    act_idx = torch.zeros((BM), dtype=torch.long, device=preds["positions"].device)
    if "target_positions" in obs_dict:
        print("DIFFUSER: WARNING using sample closest to GT from diffusion model!")
        # use the sample closest to GT
        # pred and gt may not be the same if gt is missing data at the end
        endT = min(T, obs_dict["target_positions"].size(1))
        pred_pos = preds["positions"][:,:,:endT]
        gt_pos = obs_dict["target_positions"][:,:endT].unsqueeze(1)
        gt_valid = obs_dict["target_availabilities"][:,:endT].unsqueeze(1).expand((BM, N, endT))
        err = torch.norm(pred_pos - gt_pos, dim=-1)
        err[~gt_valid] = torch.nan # so doesn't affect
        ade = torch.nanmean(err, dim=-1) # BM x N
        res_valid = torch.sum(torch.isnan(ade), dim=-1) == 0
        if torch.sum(res_valid) > 0:
            min_ade_idx = torch.argmin(ade, dim=-1)
            act_idx[res_valid] = min_ade_idx[res_valid]
    else:
        print('Could not choose sample based on GT, as no GT in data')

    return act_idx


############## GUIDANCE config ########################

class GuidanceConfig(object):
    def __init__(self, name, weight, params, agents, func=None):
        '''
        - name : name of the guidance function (i.e. the type of guidance), must be in GUIDANCE_FUNC_MAP
        - weight : alpha weight, how much affects denoising
        - params : guidance loss specific parameters
        - agents : agent indices within the scene to apply this guidance to. Applies to ALL if is None.
        - func : the function to call to evaluate this guidance loss value.
        '''
        assert name in GUIDANCE_FUNC_MAP, 'Guidance name must be one of: ' + ', '.join(map(str, GUIDANCE_FUNC_MAP.keys()))
        self.name = name
        self.weight = weight
        self.params = params
        self.agents = agents
        self.func = func

    @staticmethod
    def from_dict(config_dict):
        assert config_dict.keys() == {'name', 'weight', 'params', 'agents'}, \
                'Guidance config must include only [name, weight, params, agt_mask]. agt_mask may be None if applies to all agents in a scene'
        return GuidanceConfig(**config_dict)

    def __repr__(self):
        return '<\n%s\n>' % str('\n '.join('%s : %s' % (k, repr(v)) for (k, v) in self.__dict__.items()))

def verify_guidance_config_list(guidance_config_list):
    '''
    Returns true if there list contains some valid guidance that needs to be applied.
    Does not check to make sure each guidance dict is structured properly, only that
    the list structure is valid.
    '''
    assert len(guidance_config_list) > 0
    valid_guidance = False
    for guide in guidance_config_list:
        valid_guidance = valid_guidance or len(guide) > 0
    return valid_guidance

def verify_constraint_config(constraint_config_list):
    '''
    Given a hard constraint config dict, verifies it's structured as expected.
    Should contain fields 'agents', 'loc', and 'times'
    '''
    for constraint_config in constraint_config_list:
        if constraint_config is not None and len(constraint_config.keys()) > 0:
            assert constraint_config.keys() == {'agents', 'locs', 'times'}, \
                        'Constraint config must include only [agents, locs, times].'
            num_constraints = len(constraint_config['agents'])
            assert num_constraints == len(constraint_config['locs']), \
                        'all config fields should be same length'
            assert num_constraints == len(constraint_config['times']), \
                        'all config fields should be same length'
            if num_constraints > 0:
                assert len(constraint_config['locs'][0]) == 2, \
                    'Constraint locations must be 2d (x,y) waypoints'


############## GUIDANCE functions ########################

def apply_constraints(x, batch_scene_idx, cfg):
    '''
    Applies hard constraints to positions (x,y) specified by the given configuration.
    - x : trajectory to update with constraints. (B, N, T, D) where N is num samples and B is num agents
    - batch_scene_idx : (B,) boolean saying which scene each agent belongs to
    - cfg : list of dicts, which agents and times to apply constraints in each scene
    '''
    all_scene_inds = torch.unique_consecutive(batch_scene_idx).cpu().numpy()
    assert len(cfg) == len(all_scene_inds), "Must give the same num of configs as there are scenes in each batch"
    for i, si in enumerate(all_scene_inds):
        cur_cfg = cfg[i]
        if cur_cfg is not None and len(cur_cfg.keys()) > 0:
            cur_scene_inds = torch.nonzero(batch_scene_idx == si, as_tuple=True)[0]
            loc_tgt = torch.tensor(cur_cfg['locs']).to(x.device)
            x[cur_scene_inds[cur_cfg['agents']], :, cur_cfg['times'], :2] = loc_tgt.unsqueeze(1)
    return x

class GuidanceLoss(nn.Module):
    '''
    Abstract guidance function. This is a loss (not a reward), i.e. guidance will seek to
    MINIMIZE the implemented function.
    '''
    def __init__(self):
        super().__init__()
        self.global_t = 0

    def init_for_batch(self, example_batch):
        '''
        Initializes this loss to be used repeatedly only for the given scenes/agents in the example_batch.
        e.g. this function could use the extents of agents or num agents in each scene to cache information
              that is used while evaluating the loss
        '''
        pass

    def update(self, global_t=None):
        '''
        Update any persistant state needed by guidance loss functions.
        - global_t : the current global timestep of rollout
        '''
        if global_t is not None:
            self.global_t = global_t


    def forward(self, x, data_batch, agt_mask=None):
        '''
        Computes and returns loss value.

        Inputs:
        - x : the current trajectory (B, N, T, 6) where N is the number of samples and 6 is (x, y, vel, yaw, acc, yawvel)
        - data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        - agt_mask : size B boolean list specifying which agents to apply guidance to. Applies to ALL agents if is None.

        Output:
        - loss : (B, N) loss for each sample of each batch index. Final loss will be mean of this.
        '''
        raise NotImplementedError('Must implement guidance function evaluation')

class TargetSpeedLoss(GuidanceLoss):
    '''
    Agent should follow specific target speed.
    '''
    def __init__(self, dt, target_speed, fut_valid):
        super().__init__()
        self.target_speed = target_speed
        self.fut_valid = fut_valid
        self.dt = dt

    def forward(self, x, data_batch, agt_mask=None):
        T = x.shape[2]
        cur_tgt_speed = self.target_speed[..., self.global_t:self.global_t+T]
        fut_valid = self.fut_valid[..., self.global_t:self.global_t+T]
        
        cur_tgt_speed = torch.tensor(cur_tgt_speed, dtype=torch.float32).to(x.device)
        fut_valid = torch.tensor(fut_valid).to(x.device)

        if agt_mask is not None:
            x = x[agt_mask]
            cur_tgt_speed = cur_tgt_speed[agt_mask]
            fut_valid = fut_valid[agt_mask]

        cur_speed = x[..., 2]
        
        valid_T = cur_tgt_speed.shape[-1]
        if valid_T > 0:
            speed_dev = torch.abs(cur_speed[..., :valid_T] - cur_tgt_speed[:, None, :])
            speed_dev = torch.nan_to_num(speed_dev, nan=0)
            loss = torch.mean(speed_dev, dim=-1)
        else:
            # dummy loss
            loss = torch.mean(x, dim=[-1, -2]) * 0.
        # print('loss.shape', loss.shape)
        # print('x.shape', x.shape)
        return loss


class AgentCollisionLoss_Old(GuidanceLoss):
    '''
    Agents should not collide with each other.
    NOTE: this assumes full control over the scene. 
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x, data_batch, agt_mask=None):
        from tbsim.utils.loss_utils import collision_loss
        from tbsim.utils.l5_utils import generate_edges

        # TODO agent masking is incorrect! Want to compute collisions with ALL agents, but only store loss for desired agents.
        #       actually don't we need to stopgrad those that aren't specified? otherwise they will get an unwanted guidance.
        
        _, data_scene_index = torch.unique_consecutive(data_batch['scene_index'], return_inverse=True)
        data_type = data_batch["type"]
        data_extent = data_batch["extent"]
        data_agent_from_world = data_batch["agent_from_world"]
        data_world_from_agent = data_batch["world_from_agent"]
        data_yaw = data_batch["yaw"]
        if agt_mask is not None:
            x = x[agt_mask]
            data_scene_index = data_scene_index[agt_mask]
            data_type = data_type[agt_mask]
            data_extent = data_extent[agt_mask]
            data_agent_from_world = data_agent_from_world[agt_mask]
            data_world_from_agent = data_world_from_agent[agt_mask]
            data_yaw = data_yaw[agt_mask]

        pos_pred = x[..., :2]
        yaw_pred = x[..., 3:4]

        pos_pred_global, yaw_pred_global = transform_agents_to_world(pos_pred, yaw_pred, data_world_from_agent)

        loss = 0.0
        scene_inds = torch.unique_consecutive(data_scene_index)
        for scene_idx in scene_inds:
            cur_scene_mask = data_scene_index == scene_idx

            # NOTE: we assume each sample is a new version of the scene,
            #       so we will apply the collision loss to make sure agents in just this sample don't collide.
            #       Generate edges assumes each scene indexing dimension is 0, so we 
            #       will treat each sample as a different "scene" for this.
            scene_pos = pos_pred_global[cur_scene_mask].transpose(0, 1) # [num_samp, num_agt, timesteps, 2]
            scene_yaw = yaw_pred_global[cur_scene_mask].transpose(0, 1)

            raw_type = data_type[cur_scene_mask].unsqueeze(0).expand((scene_pos.size(0), scene_pos.size(1)))
            extents = data_extent[cur_scene_mask][:, :2].unsqueeze(0).expand((scene_pos.size(0), scene_pos.size(1), 2))

            edges = generate_edges(raw_type, extents, scene_pos, scene_yaw)
            # NOTE this will be a scalar instead of BxN as desired
            loss = loss + collision_loss(edges) # loss already goes through sigmoid before

        return loss

class AgentCollisionLoss_Slow(GuidanceLoss):
    '''
    Agents should not collide with each other.
    NOTE: this assumes full control over the scene. 
    '''
    def __init__(self, num_disks=5, buffer_dist=0.2):
        '''
        - buffer_len : additional space to leave between agents
        '''
        super().__init__()
        # TODO could pre-cache the disks if given "extents" ahead of time
        # TODO could also pre-cache batch/scene masking
        self.num_disks = num_disks
        self.buffer_dist = buffer_dist

    def init_disks(self, num_disks, extents):
        NA = extents.size(0)
        agt_rad = extents[:, 1] / 2. # assumes lenght > width
        cent_min = -(extents[:, 0] / 2.) + agt_rad
        cent_max = (extents[:, 0] / 2.) - agt_rad
        # sample disk centroids along x axis
        cent_x = torch.stack([torch.linspace(cent_min[vidx].item(), cent_max[vidx].item(), num_disks) \
                                for vidx in range(NA)], dim=0).to(extents.device)
        centroids = torch.stack([cent_x, torch.zeros_like(cent_x)], dim=2)      
        return centroids, agt_rad

    def forward(self, x, data_batch, agt_mask=None):   
        _, data_scene_index = torch.unique_consecutive(data_batch['scene_index'], return_inverse=True)
        data_extent = data_batch["extent"]
        data_world_from_agent = data_batch["world_from_agent"]

        pos_pred = x[..., :2]
        yaw_pred = x[..., 3:4]

        pos_pred_global, yaw_pred_global = transform_agents_to_world(pos_pred, yaw_pred, data_world_from_agent)
        if agt_mask is not None:
            # only want gradient to backprop to agents being guided
            pos_pred_detach = pos_pred_global.clone().detach()
            yaw_pred_detach = yaw_pred_global.clone().detach()

            pos_pred_global = torch.where(agt_mask[:,None,None,None].expand_as(pos_pred_global),
                                          pos_pred_global,
                                          pos_pred_detach)
            yaw_pred_global = torch.where(agt_mask[:,None,None,None].expand_as(yaw_pred_global),
                                          yaw_pred_global,
                                          yaw_pred_detach)

        # create disks and transform to world frame (centroids)
        B, N, T, _ = pos_pred_global.size()
        BN = B*N
        centroids, agt_rad = self.init_disks(self.num_disks, data_extent) # B x num_disks x 2
        centroids = centroids[:,None,None].expand(B, N, T, self.num_disks, 2)
        agt_rad = agt_rad[:,None].expand(B, N)
        # to world
        s = torch.sin(yaw_pred_global).unsqueeze(-1)
        c = torch.cos(yaw_pred_global).unsqueeze(-1)
        rotM = torch.cat((torch.cat((c, s), dim=-1), torch.cat((-s, c), dim=-1)), dim=-2)
        centroids = torch.matmul(centroids, rotM) + pos_pred_global.unsqueeze(-2)

        # NOTE: debug viz sanity check
        # import matplotlib
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # for ni in range(centroids.size(0)):
        #     plt.plot(centroids[ni,0,:,2,0].detach().cpu().numpy(),
        #              centroids[ni,0,:,2,1].detach().cpu().numpy(),
        #              '-')
        # plt.gca().set_xlim([15, -15])
        # plt.gca().set_ylim([-15, 15])
        # plt.show()
        # plt.close(fig)

        # NOTE: assume each sample is a different scene for the sake of computing collisions
        centroids = centroids.transpose(0,1).reshape((BN, T, self.num_disks, 2)) # N*B x T x D x 2
        agt_rad = agt_rad.transpose(0,1).reshape((BN)) # N*B
        # minimum distance that two vehicle circle centers can be apart without collision (+ buffer)
        penalty_dists = agt_rad.view(BN, 1).expand(BN, BN) + agt_rad.view(1, BN).expand(BN, BN) + self.buffer_dist

        # build mask for which agents to actually compare to each other
        #       only want to penalize collisions with vehicles in the same scene (and same sample)
        # each sample mask is a block diagonal for the contained scenes
        sample_block_list = []
        scene_inds = torch.unique_consecutive(data_scene_index)
        for scene_idx in scene_inds:
            cur_scene_mask = data_scene_index == scene_idx
            num_agt_in_scene = torch.sum(cur_scene_mask)
            sample_block_list.append(torch.ones((num_agt_in_scene, num_agt_in_scene), dtype=torch.bool))
        sample_block = torch.block_diag(*sample_block_list).to(centroids.device)
        # now build full mask matrix
        scene_blocks = sample_block.unsqueeze(0).expand((N, B, B))
        scene_mask = torch.block_diag(*scene_blocks)
        # don't want to compute self collisions
        scene_mask = torch.logical_and(scene_mask, ~torch.eye(BN, dtype=torch.bool).to(centroids.device))

        centroids = centroids.transpose(0, 1) # T x N*B X D x 2
        # distances between all pairs of circles between all pairs of agents
        cur_cent1 = centroids.view(T, BN, 1, self.num_disks, 2).expand(T, BN, BN, self.num_disks, 2).reshape(T*BN*BN, self.num_disks, 2)
        cur_cent2 = centroids.view(T, 1, BN, self.num_disks, 2).expand(T, BN, BN, self.num_disks, 2).reshape(T*BN*BN, self.num_disks, 2)
        pair_dists = torch.cdist(cur_cent1, cur_cent2).view(T*BN*BN, self.num_disks*self.num_disks)
        # get minimum distance over all circle pairs between each pair of agents
        pair_dists = torch.min(pair_dists, 1)[0].view(T, BN, BN)

        penalty_dists = penalty_dists.view(1, BN, BN)
        is_colliding_mask = torch.logical_and(pair_dists <= penalty_dists,
                                              scene_mask.view(1, BN, BN))

        # penalty is inverse normalized distance apart for those already colliding
        cur_penalties = 1.0 - (pair_dists / penalty_dists)
        # only compute loss where it's valid and colliding
        cur_penalties = torch.where(is_colliding_mask,
                                    cur_penalties,
                                    torch.zeros_like(cur_penalties))
                                        
        # summing over timesteps and all other agents to get N*B -> B x N
        cur_penalties = cur_penalties.sum(0).sum(1).reshape((N,B)).transpose(0,1)

        # print(cur_penalties)
        if agt_mask is not None:
            return cur_penalties[agt_mask]
        else:
            return cur_penalties

#
# TODO this is not very efficient for num_scene_in_batch > 1
#       since there will be two different agent collision losses, both of which
#       compute the same thing just mask it differently. Really should apply 
#       agent mask before computing anything, but this does not work if
#       the agent_collision is only being applied to a subset of one scene.
#
class AgentCollisionLoss(GuidanceLoss):
    '''
    Agents should not collide with each other.
    NOTE: this assumes full control over the scene. 
    '''
    def __init__(self, num_disks=5, buffer_dist=0.2, decay_rate=0.9, guide_moving_speed_th=5e-1, excluded_agents=None):
        '''
        - num_disks : the number of disks to use to approximate the agent for collision detection.
                        more disks improves accuracy
                        num_disks 
                         num_disks  5 5 
        - buffer_dist : additional space to leave between agents
        - decay_rate : how much to decay the loss as time goes on
        - excluded_agents : the collisions among these agents will not be penalized
        '''
        super().__init__()
        self.num_disks = num_disks
        self.buffer_dist = buffer_dist
        self.decay_rate = decay_rate
        self.guide_moving_speed_th = guide_moving_speed_th

        self.centroids = None
        self.penalty_dists = None
        self.scene_mask = None
        self.excluded_agents = excluded_agents

    def init_for_batch(self, example_batch):
        '''
        Caches disks and masking ahead of time.
        '''
        # return 
        # pre-compute disks to approximate each agent
        data_extent = example_batch["extent"]
        self.centroids, agt_rad = self.init_disks(self.num_disks, data_extent) # B x num_disks x 2
        B = self.centroids.size(0)
        # minimum distance that two vehicle circle centers can be apart without collision (+ buffer)
        self.penalty_dists = agt_rad.view(B, 1).expand(B, B) + agt_rad.view(1, B).expand(B, B) + self.buffer_dist
        
        # pre-compute masking for vectorized pairwise distance computation
        self.scene_mask = self.init_mask(example_batch['scene_index'], self.centroids.device)

    def init_disks(self, num_disks, extents):
        NA = extents.size(0)
        agt_rad = extents[:, 1] / 2. # assumes length > width
        cent_min = -(extents[:, 0] / 2.) + agt_rad
        cent_max = (extents[:, 0] / 2.) - agt_rad
        # sample disk centroids along x axis
        cent_x = torch.stack([torch.linspace(cent_min[vidx].item(), cent_max[vidx].item(), num_disks) \
                                for vidx in range(NA)], dim=0).to(extents.device)
        centroids = torch.stack([cent_x, torch.zeros_like(cent_x)], dim=2)      
        return centroids, agt_rad

    # TODO why are results when using num_scenes_per_batch > 1 different than = 1?
    def init_mask(self, batch_scene_index, device):
        _, data_scene_index = torch.unique_consecutive(batch_scene_index, return_inverse=True)
        scene_block_list = []
        scene_inds = torch.unique_consecutive(data_scene_index)
        for scene_idx in scene_inds:
            cur_scene_mask = data_scene_index == scene_idx
            num_agt_in_scene = torch.sum(cur_scene_mask)
            cur_scene_block = ~torch.eye(num_agt_in_scene, dtype=torch.bool)
            scene_block_list.append(cur_scene_block)
        scene_mask = torch.block_diag(*scene_block_list).to(device)
        return scene_mask

    def forward(self, x, data_batch, agt_mask=None):
        data_extent = data_batch["extent"]
        data_world_from_agent = data_batch["world_from_agent"]
        curr_speed = data_batch['curr_speed']
        scene_index = data_batch['scene_index']

        # consider collision gradients only for those moving vehicles
        moving = torch.abs(curr_speed) > self.guide_moving_speed_th
        stationary = ~moving
        stationary = stationary.view(-1, 1, 1, 1).expand_as(x)
        x_new = x.clone()
        x_new[stationary] = x[stationary].detach()
        x = x_new

        pos_pred = x[..., :2]
        yaw_pred = x[..., 3:4]

        pos_pred_global, yaw_pred_global = transform_agents_to_world(pos_pred, yaw_pred, data_world_from_agent)
        if agt_mask is not None:
            # only want gradient to backprop to agents being guided
            pos_pred_detach = pos_pred_global.detach().clone()
            yaw_pred_detach = yaw_pred_global.detach().clone()

            pos_pred_global = torch.where(agt_mask[:,None,None,None].expand_as(pos_pred_global),
                                          pos_pred_global,
                                          pos_pred_detach)
            yaw_pred_global = torch.where(agt_mask[:,None,None,None].expand_as(yaw_pred_global),
                                          yaw_pred_global,
                                          yaw_pred_detach)

        # create disks and transform to world frame (centroids)
        B, N, T, _ = pos_pred_global.size()
        if self.centroids is None or self.penalty_dists is None:
            centroids, agt_rad = self.init_disks(self.num_disks, data_extent) # B x num_disks x 2
            # minimum distance that two vehicle circle centers can be apart without collision (+ buffer)
            penalty_dists = agt_rad.view(B, 1).expand(B, B) + agt_rad.view(1, B).expand(B, B) + self.buffer_dist
        else:
            centroids, penalty_dists = self.centroids, self.penalty_dists
        centroids = centroids[:,None,None].expand(B, N, T, self.num_disks, 2)
        # to world
        s = torch.sin(yaw_pred_global).unsqueeze(-1)
        c = torch.cos(yaw_pred_global).unsqueeze(-1)
        rotM = torch.cat((torch.cat((c, s), dim=-1), torch.cat((-s, c), dim=-1)), dim=-2)
        centroids = torch.matmul(centroids, rotM) + pos_pred_global.unsqueeze(-2)

        # NOTE: debug viz sanity check
        # import matplotlib
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # for ni in range(centroids.size(0)):
        #     plt.plot(centroids[ni,0,:,2,0].detach().cpu().numpy(),
        #              centroids[ni,0,:,2,1].detach().cpu().numpy(),
        #              '-')
        # plt.gca().set_xlim([15, -15])
        # plt.gca().set_ylim([-15, 15])
        # plt.show()
        # plt.close(fig)

        # NOTE: assume each sample is a different scene for the sake of computing collisions
        if self.scene_mask is None:
            scene_mask = self.init_mask(scene_index, centroids.device)
        else:
            scene_mask = self.scene_mask

        # TODO technically we do not need all BxB comparisons
        #       only need the lower triangle of this matrix (no self collisions and only one way distance)
        #       but this may be slower to assemble than masking

        # TODO B could contain multiple scenes, could just pad each scene to the max_agents and compare MaxA x MaxA to avoid unneeded comparisons across scenes

        centroids = centroids.transpose(0,2) # T x NS x B x D x 2
        centroids = centroids.reshape((T*N, B, self.num_disks, 2))
        # distances between all pairs of circles between all pairs of agents
        cur_cent1 = centroids.view(T*N, B, 1, self.num_disks, 2).expand(T*N, B, B, self.num_disks, 2).reshape(T*N*B*B, self.num_disks, 2)
        cur_cent2 = centroids.view(T*N, 1, B, self.num_disks, 2).expand(T*N, B, B, self.num_disks, 2).reshape(T*N*B*B, self.num_disks, 2)
        pair_dists = torch.cdist(cur_cent1, cur_cent2).view(T*N*B*B, self.num_disks*self.num_disks)#num_disks
        # get minimum distance over all circle pairs between each pair of agents
        pair_dists = torch.min(pair_dists, 1)[0].view(T*N, B, B)

        penalty_dists = penalty_dists.view(1, B, B)
        is_colliding_mask = torch.logical_and(pair_dists <= penalty_dists,
                                              scene_mask.view(1, B, B))#scene_mask,,
        
        if self.excluded_agents is not None:
            # for all row and column pairs that are both in the excluded agents list, set to 0
            excluded_agents_mask = torch.ones((1, B, B), device=is_colliding_mask.device)
            excluded_agents_tensor = torch.tensor(self.excluded_agents, device=is_colliding_mask.device)
            i_indices, j_indices = torch.meshgrid(excluded_agents_tensor, excluded_agents_tensor, indexing='ij')
            excluded_agents_mask[0, i_indices, j_indices] = 0    

            is_colliding_mask = torch.logical_and(is_colliding_mask, excluded_agents_mask)
        
        # # consider collision only for those involving at least one vehicle moving
        # moving = torch.abs(data_batch['curr_speed']) > self.guide_moving_speed_th
        # moving1 = moving.view(1, B, 1).expand(1, B, B)
        # moving2 = moving.view(1, 1, B).expand(1, B, B)
        # moving_mask = torch.logical_or(moving1, moving2) 
        # is_colliding_mask = torch.logical_and(is_colliding_mask,
        #                                       moving_mask)

        # penalty is inverse normalized distance apart for those already colliding
        cur_penalties = 1.0 - (pair_dists / penalty_dists)
        # only compute loss where it's valid and colliding
        cur_penalties = torch.where(is_colliding_mask,
                                    cur_penalties,
                                    torch.zeros_like(cur_penalties))
                                        
        # summing over timesteps and all other agents to get B x N
        cur_penalties = cur_penalties.reshape((T, N, B, B))
        # cur_penalties = cur_penalties.sum(0).sum(-1).transpose(0, 1)
        # penalize early steps more than later steps
        exp_weights = torch.tensor([self.decay_rate ** t for t in range(T)], device=cur_penalties.device)
        exp_weights /= exp_weights.sum()
        cur_penalties = cur_penalties * exp_weights[:, None, None, None]
        cur_penalties = cur_penalties.sum(0).mean(-1).transpose(0, 1)

        # consider loss only for those agents that are moving (note: since the loss involves interaction those stationary vehicles will still be indirectly penalized from the loss of other moving vehicles)
        cur_penalties = torch.where(moving.unsqueeze(-1).expand(B, N), cur_penalties, torch.zeros_like(cur_penalties))

        # print(cur_penalties)
        if agt_mask is not None:
            return cur_penalties[agt_mask]
        else:
            return cur_penalties
class EgoEscapeLoss(GuidanceLoss):
    '''
    Agents should not collide with each other.
    NOTE: this assumes full control over the scene. 
    '''
    def __init__(self,target_ind=2, ref_ind=3, num_disks=5, buffer_dist=0.2, decay_rate=0.9, guide_moving_speed_th=5e-1, excluded_agents=None,collision_type=None,scope=52,GT=False):
        '''
        - num_disks : the number of disks to use to approximate the agent for collision detection.
                        more disks improves accuracy
                        num_disks 
                         num_disks  5 5 
        - buffer_dist : additional space to leave between agents
        - decay_rate : how much to decay the loss as time goes on
        - excluded_agents : the collisions among these agents will not be penalized
        '''
        super().__init__()
        self.num_disks = num_disks
        self.buffer_dist = buffer_dist
        self.decay_rate = decay_rate
        self.guide_moving_speed_th = guide_moving_speed_th

        self.centroids = None
        self.penalty_dists = None
        self.scene_mask = None
        self.excluded_agents = excluded_agents
        self.scope=scope
        self.GT=GT

    def init_for_batch(self, example_batch):
        '''
        Caches disks and masking ahead of time.
        '''
        # return 
        # pre-compute disks to approximate each agent
        data_extent = example_batch["extent"]
        self.centroids, agt_rad = self.init_disks(self.num_disks, data_extent) # B x num_disks x 2
        B = self.centroids.size(0)
        # minimum distance that two vehicle circle centers can be apart without collision (+ buffer)
        self.penalty_dists = agt_rad.view(B, 1).expand(B, B) + agt_rad.view(1, B).expand(B, B) + self.buffer_dist
        
        # pre-compute masking for vectorized pairwise distance computation
        self.scene_mask = self.init_mask(example_batch['scene_index'], self.centroids.device)

    def init_disks(self, num_disks, extents):
        NA = extents.size(0)
        agt_rad = extents[:, 1] / 2. # assumes length > width
        cent_min = -(extents[:, 0] / 2.) + agt_rad
        cent_max = (extents[:, 0] / 2.) - agt_rad
        # sample disk centroids along x axis
        cent_x = torch.stack([torch.linspace(cent_min[vidx].item(), cent_max[vidx].item(), num_disks) \
                                for vidx in range(NA)], dim=0).to(extents.device)
        centroids = torch.stack([cent_x, torch.zeros_like(cent_x)], dim=2)      
        return centroids, agt_rad

    # TODO why are results when using num_scenes_per_batch > 1 different than = 1?
    def init_mask(self, batch_scene_index, device):
        _, data_scene_index = torch.unique_consecutive(batch_scene_index, return_inverse=True)
        scene_block_list = []
        scene_inds = torch.unique_consecutive(data_scene_index)
        for scene_idx in scene_inds:
            cur_scene_mask = data_scene_index == scene_idx
            num_agt_in_scene = torch.sum(cur_scene_mask)
            cur_scene_block = ~torch.eye(num_agt_in_scene, dtype=torch.bool)
            scene_block_list.append(cur_scene_block)
        scene_mask = torch.block_diag(*scene_block_list).to(device)
        return scene_mask

    def forward(self, x, data_batch, agt_mask=None):
        data_extent = data_batch["extent"]
        data_world_from_agent = data_batch["world_from_agent"]
        curr_speed = data_batch['curr_speed']
        scene_index = data_batch['scene_index']

        # consider collision gradients only for those moving vehicles
        # moving = torch.abs(curr_speed) > self.guide_moving_speed_th
        # stationary = ~moving
        # stationary = stationary.view(-1, 1, 1, 1).expand_as(x)
        # x_new = x.clone()
        # x_new[stationary] = x[stationary].detach()
        # x = x_new

        pos_pred = x[..., :2]
        yaw_pred = x[..., 3:4]

        
        
            #poscut


        pos_pred_global, yaw_pred_global = transform_agents_to_world(pos_pred, yaw_pred, data_world_from_agent)
         #ground truth
        if self.GT:
            first_time_world_positions =data_batch['first_time_world_positions']
            first_time_world_yaws =data_batch['first_time_world_yaws']
        
            if first_time_world_positions.shape[1] < pos_pred.shape[2]:
                pos_pred_global= pos_pred_global[:,:,:first_time_world_positions.shape[1],:]
                yaw_pred_global= yaw_pred_global[:,:,:first_time_world_positions.shape[1],:]
            # pos_pred_global[1:,:,:]=first_time_world_positions[1:,:].unsqueeze(1).repeat(1, pos_pred.shape[1], 1,1)
            min_size = min(pos_pred_global.shape[0]-1, first_time_world_positions.shape[0]-1)
            pos_pred_global[1:min_size+1,:,:] = first_time_world_positions[1:min_size+1,:].unsqueeze(1).repeat(1, pos_pred.shape[1], 1,1)
            # yaw_pred_global[1:,:,:]=first_time_world_yaws[1:,:].unsqueeze(1).repeat(1, pos_pred.shape[1], 1,1)
            yaw_pred_global[1:min_size+1,:,:] = first_time_world_yaws[1:min_size+1,:].unsqueeze(1).repeat(1, pos_pred.shape[1], 1,1)
        if agt_mask is not None:
            # only want gradient to backprop to agents being guided
            pos_pred_detach = pos_pred_global.detach().clone()
            yaw_pred_detach = yaw_pred_global.detach().clone()

            pos_pred_global = torch.where(agt_mask[:,None,None,None].expand_as(pos_pred_global),
                                          pos_pred_global,
                                          pos_pred_detach)
            yaw_pred_global = torch.where(agt_mask[:,None,None,None].expand_as(yaw_pred_global),
                                          yaw_pred_global,
                                          yaw_pred_detach)

        # create disks and transform to world frame (centroids)
        B, N, T, _ = pos_pred_global.size()
        if self.centroids is None or self.penalty_dists is None:
            centroids, agt_rad = self.init_disks(self.num_disks, data_extent) # B x num_disks x 2
            # minimum distance that two vehicle circle centers can be apart without collision (+ buffer)
            penalty_dists = agt_rad.view(B, 1).expand(B, B) + agt_rad.view(1, B).expand(B, B) + self.buffer_dist
        else:
            centroids, penalty_dists = self.centroids, self.penalty_dists
        centroids = centroids[:,None,None].expand(B, N, T, self.num_disks, 2)
        # to world
        s = torch.sin(yaw_pred_global).unsqueeze(-1)
        c = torch.cos(yaw_pred_global).unsqueeze(-1)
        rotM = torch.cat((torch.cat((c, s), dim=-1), torch.cat((-s, c), dim=-1)), dim=-2)
        centroids = torch.matmul(centroids, rotM) + pos_pred_global.unsqueeze(-2)

        # NOTE: debug viz sanity check
        # import matplotlib
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # for ni in range(centroids.size(0)):
        #     plt.plot(centroids[ni,0,:,2,0].detach().cpu().numpy(),
        #              centroids[ni,0,:,2,1].detach().cpu().numpy(),
        #              '-')
        # plt.gca().set_xlim([15, -15])
        # plt.gca().set_ylim([-15, 15])
        # plt.show()
        # plt.close(fig)

        # NOTE: assume each sample is a different scene for the sake of computing collisions
        if self.scene_mask is None:
            scene_mask = self.init_mask(scene_index, centroids.device)
        else:
            scene_mask = self.scene_mask

        # TODO technically we do not need all BxB comparisons
        #       only need the lower triangle of this matrix (no self collisions and only one way distance)
        #       but this may be slower to assemble than masking

        # TODO B could contain multiple scenes, could just pad each scene to the max_agents and compare MaxA x MaxA to avoid unneeded comparisons across scenes

        centroids = centroids.transpose(0,2) # T x NS x B x D x 2
        centroids = centroids.reshape((T*N, B, self.num_disks, 2))
        # distances between all pairs of circles between all pairs of agents
        cur_cent1 = centroids.view(T*N, B, 1, self.num_disks, 2).expand(T*N, B, B, self.num_disks, 2).reshape(T*N*B*B, self.num_disks, 2)
        cur_cent2 = centroids.view(T*N, 1, B, self.num_disks, 2).expand(T*N, B, B, self.num_disks, 2).reshape(T*N*B*B, self.num_disks, 2)
        pair_dists = torch.cdist(cur_cent1, cur_cent2).view(T*N*B*B, self.num_disks*self.num_disks)#num_disks
        # get minimum distance over all circle pairs between each pair of agents
        pair_dists = torch.min(pair_dists, 1)[0].view(T*N, B, B)

        penalty_dists = penalty_dists.view(1, B, B)
        is_colliding_mask = torch.logical_and(pair_dists <= penalty_dists,
                                              scene_mask.view(1, B, B))#scene_mask,,
        
        if self.excluded_agents is not None:
            # for all row and column pairs that are both in the excluded agents list, set to 0
            excluded_agents_mask = torch.ones((1, B, B), device=is_colliding_mask.device)
            excluded_agents_tensor = torch.tensor(self.excluded_agents, device=is_colliding_mask.device)
            i_indices, j_indices = torch.meshgrid(excluded_agents_tensor, excluded_agents_tensor, indexing='ij')
            excluded_agents_mask[0, i_indices, j_indices] = 0    

            is_colliding_mask = torch.logical_and(is_colliding_mask, excluded_agents_mask)
        
        # # consider collision only for those involving at least one vehicle moving
        # moving = torch.abs(data_batch['curr_speed']) > self.guide_moving_speed_th
        # moving1 = moving.view(1, B, 1).expand(1, B, B)
        # moving2 = moving.view(1, 1, B).expand(1, B, B)
        # moving_mask = torch.logical_or(moving1, moving2) 
        # is_colliding_mask = torch.logical_and(is_colliding_mask,
        #                                       moving_mask)

        # penalty is inverse normalized distance apart for those already colliding
        cur_penalties = 1.0 - (pair_dists / penalty_dists)
        # only compute loss where it's valid and colliding
        cur_penalties = torch.where(is_colliding_mask,
                                    cur_penalties,
                                    torch.zeros_like(cur_penalties))
        
        #scope1
        
        # summing over timesteps and all other agents to get B x N
        cur_penalties = cur_penalties.reshape((T, N, B, B))
        time_scope_mask = torch.zeros((T, 1, 1, 1), device=cur_penalties.device)
        if T > 0:  # 
            scope_limit = min(self.scope, T)  # 
            time_scope_mask[:scope_limit] = 1.0  #  self.scope 1

        # 
        cur_penalties = cur_penalties * time_scope_mask
        # cur_penalties = cur_penalties.sum(0).sum(-1).transpose(0, 1)
        # penalize early steps more than later steps
        exp_weights = torch.tensor([self.decay_rate ** t for t in range(T)], device=cur_penalties.device)
        exp_weights /= exp_weights.sum()
        cur_penalties = cur_penalties * exp_weights[:, None, None, None]
        cur_penalties = cur_penalties.sum(0).mean(-1).transpose(0, 1)
        

        # consider loss only for those agents that are moving (note: since the loss involves interaction those stationary vehicles will still be indirectly penalized from the loss of other moving vehicles)
        # cur_penalties = torch.where(moving.unsqueeze(-1).expand(B, N), cur_penalties, torch.zeros_like(cur_penalties))
        

        # print(cur_penalties)
        if agt_mask is not None:
            return cur_penalties[agt_mask]
        else:
            return cur_penalties

class CollisionLoss_disk(GuidanceLoss):
    '''
    Vehicle with index target_ind should collide with vehicle with index ref_ind.
    Using disk-based collision detection for more accurate results.
    '''
    def __init__(self, target_ind=2, ref_ind=3, num_disks=5, buffer_dist=0.2, decay_rate=0.9,collision_type=None):
        '''
        - target_ind: index of the target vehicle that should collide
        - ref_ind: index of the reference vehicle to collide with
        - num_disks: number of disks to approximate each agent's shape
        - buffer_dist: additional distance buffer for collision detection
        - decay_rate: how much to decay the loss as time goes on (0-1)
                     lower values place more emphasis on earlier timesteps
        '''
        super().__init__()
        self.target_ind = target_ind
        self.ref_ind = ref_ind
        self.num_disks = num_disks
        self.buffer_dist = buffer_dist
        self.decay_rate = decay_rate
        self.collision_type = collision_type
        
    def init_for_batch(self, example_batch):
        '''
        Caches disks and masking ahead of time.
        '''
        # return 
        # pre-compute disks to approximate each agent
        data_extent = example_batch["extent"]
        self.centroids, agt_rad = self.init_disks(self.num_disks, data_extent) # B x num_disks x 2
        B = self.centroids.size(0)
        # minimum distance that two vehicle circle centers can be apart without collision (+ buffer)
        self.penalty_dists = agt_rad.view(B, 1).expand(B, B) + agt_rad.view(1, B).expand(B, B) + self.buffer_dist
        
        # pre-compute masking for vectorized pairwise distance computation
        self.scene_mask = self.init_mask(example_batch['scene_index'], self.centroids.device)

    def init_disks(self, num_disks, extents):
        NA = extents.size(0)
        agt_rad = extents[:, 1] / 2. # assumes length > width
        cent_min = -(extents[:, 0] / 2.) + agt_rad
        cent_max = (extents[:, 0] / 2.) - agt_rad
        # sample disk centroids along x axis
        cent_x = torch.stack([torch.linspace(cent_min[vidx].item(), cent_max[vidx].item(), num_disks) \
                                for vidx in range(NA)], dim=0).to(extents.device)
        centroids = torch.stack([cent_x, torch.zeros_like(cent_x)], dim=2)      
        return centroids, agt_rad

    # TODO why are results when using num_scenes_per_batch > 1 different than = 1?
    def init_mask(self, batch_scene_index, device):
        _, data_scene_index = torch.unique_consecutive(batch_scene_index, return_inverse=True)
        scene_block_list = []
        scene_inds = torch.unique_consecutive(data_scene_index)
        for scene_idx in scene_inds:
            cur_scene_mask = data_scene_index == scene_idx
            num_agt_in_scene = torch.sum(cur_scene_mask)
            cur_scene_block = ~torch.eye(num_agt_in_scene, dtype=torch.bool)
            scene_block_list.append(cur_scene_block)
        scene_mask = torch.block_diag(*scene_block_list).to(device)
        return scene_mask

    def forward(self, x, data_batch, agt_mask=None,**kwargs):
        B, N, T, _ = x.shape
        true_indices = torch.nonzero(agt_mask).squeeze()
        target_ind = true_indices[self.target_ind].item()
        ref_ind = true_indices[self.ref_ind].item()
        # Get agent extents (dimensions)
        data_extent = data_batch["extent"]
        data_world_from_agent = data_batch["world_from_agent"]
        
        # Extract positions and yaws
        pos_pred = x[..., :2]
        yaw_pred = x[..., 3:4]
        
        # Transform to world coordinates
        pos_pred_global, yaw_pred_global = transform_agents_to_world(pos_pred, yaw_pred, data_world_from_agent)
        
        pos_pred_global[target_ind] = pos_pred_global[target_ind].detach().clone()
        yaw_pred_global[target_ind] = yaw_pred_global[target_ind].detach().clone()
        # Initialize disks for agents
        if self.centroids is None or self.penalty_dists is None:
            centroids, agt_rad = self.init_disks(self.num_disks, data_extent) # B x num_disks x 2
            # minimum distance that two vehicle circle centers can be apart without collision (+ buffer)
            penalty_dists = agt_rad.view(B, 1).expand(B, B) + agt_rad.view(1, B).expand(B, B) + self.buffer_dist
        else:
            centroids, penalty_dists = self.centroids, self.penalty_dists

        # Transform centroids to world coordinates
        centroids = centroids[:,None,None].expand(B, N, T, self.num_disks, 2)
        # Rotate based on yaw
        s = torch.sin(yaw_pred_global).unsqueeze(-1)
        c = torch.cos(yaw_pred_global).unsqueeze(-1)
        rotM = torch.cat((torch.cat((c, s), dim=-1), torch.cat((-s, c), dim=-1)), dim=-2)
        centroids = torch.matmul(centroids, rotM) + pos_pred_global.unsqueeze(-2)
        # agenet_masktrueself.target_indself.target_ind
        

                
        # Select only the target and reference agents
        centroids_i = select_agent_ind(centroids, target_ind).detach().clone()  # (N, T, num_disks, 2)
        centroids_j = select_agent_ind(centroids, ref_ind)    # (N, T, num_disks, 2)
        
        
        # Minimum distance for collision detection (sum of radii + buffer)
        min_dist = penalty_dists[target_ind, ref_ind]
        
        # Compute pairwise distances between all disks of the two agents
        # Reshape for pairwise distance computation
        centroids_i_flat = centroids_i.reshape(N*T, self.num_disks, 2)
        centroids_j_flat = centroids_j.reshape(N*T, self.num_disks, 2)
        
        # Calculate distances between all disk pairs
        pair_dists = torch.cdist(centroids_i_flat, centroids_j_flat)  # (N*T, num_disks, num_disks)
        
        # Get minimum distance over all circle pairs
        min_pair_dists = torch.min(pair_dists.view(N*T, self.num_disks*self.num_disks), dim=1)[0]
        min_pair_dists = min_pair_dists.view(N, T)
        
        # Check if vehicles are colliding (distance <= min_dist)
        is_colliding = min_pair_dists <= min_dist
        
        # For collision loss, we want to ENCOURAGE collision, so penalize when NOT colliding
        # We need to invert the usual collision avoidance loss
        collision_loss = torch.where(
            is_colliding,
            torch.zeros_like(min_pair_dists),           # No penalty when colliding
            min_pair_dists                  # Penalty proportional to distance when not colliding
        )
        
        # Apply decay rate to weight earlier timesteps more heavily
        # Calculate temporal weights based on decay rate
        time_weights = torch.tensor([self.decay_rate ** t for t in range(T)], device=collision_loss.device)
        time_weights = time_weights / time_weights.sum()  # Normalize weights
        
        # Apply weights to the loss at each timestep
        weighted_loss = collision_loss * time_weights.view(1, T)
        
        # Sum over time (instead of mean, as we've already weighted)
        
        loss = weighted_loss.mean(-1)
        # loss = torch.zeros_like(weighted_loss.mean(-1))
        return loss
        



class CollisionLoss(GuidanceLoss):
    '''
    Vehicle with index target_ind should collide with vehicle with index ref_ind.
    '''
    def __init__(self, target_ind=2, ref_ind=3, collision_radius=1.0,**kwargs):
        super().__init__()
        self.target_ind = target_ind
        self.ref_ind = ref_ind
        self.collision_radius = collision_radius

    def forward(self, x, data_batch, agt_mask=None,**kwargs):
        B, N, T, _ = x.shape
        if agt_mask is not None:
            x = x[agt_mask]
        # Select positions
        # (B,N,T,6) -> (B,N,T,2)
        pos_pred = x[..., :2]
        # Select yaws
        # (B,N,T,6) -> (B,N,T,1)
        yaw_pred = x[..., 3:4]
        # convert prediction from the respective agent coordinates to the world coordinate
        # (B,N,T,2), (B,N,T,1), dict -> (B,N,T,2), (B,N,T,1)
        pos_pred_world, yaw_pred_world = transform_coord_agents_to_world(pos_pred, yaw_pred, data_batch)

        # select the relevant agents with index self.target_ind in the world coordinate
        # (B, N, T, 2), int -> (N, T, 2)
        pos_pred_i_world = select_agent_ind(pos_pred_world, self.target_ind).detach().clone()
        # select the relevant agents with index self.ref_ind in the world coordinate
        # (B, N, T, 2), int -> (N, T, 2)
        pos_pred_j_world = select_agent_ind(pos_pred_world, self.ref_ind)

        # Compute the distance between the two vehicles
        # (N, T, 2), (N, T, 2) -> (N, T)
        dist = torch.norm(pos_pred_i_world - pos_pred_j_world, dim=-1)

        # Compute the collision loss by penalizing the distance greater than the collision radius
        # (N, T) -> (N, T)
        collision_loss = torch.clip(dist - self.collision_radius, min=0)

        # Take the mean over time
        # (N, T) -> (N)
        loss = collision_loss.mean(-1)

        return loss




# TODO loss
#

# egoyaw

# x0yaw180

# cut inx0yaw45-45

# lossx<0,yaw0

# lossx>0,yaw0

# T-bone attacklossyaw90-90x0
# class CollisionType(IntEnum):
#     """This enum defines the three types of collisions: front, rear and side."""
#     FRONT = 0
#     REAR = 1
#     SIDE = 2
class CollisionTypeLoss(GuidanceLoss):
    def __init__(self, target_ind=2, ref_ind=3, collision_radius=2.0, collision_type='A vehicle cuts in and collides with the ego vehicle'):
        super().__init__()
        self.target_ind = target_ind
        self.ref_ind = ref_ind
        self.collision_radius = collision_radius
        self.collision_type = collision_type
        self.yaw_collision_type_dict = {
            'A vehicle cuts in and collides with the ego vehicle': [(-torch.pi/4, torch.pi/4)],
            'A vehicle rear-ends the ego vehicle': [(-torch.pi/4,torch.pi/4)],
            'Ego vehicle rear-ends another vehicle': [(-torch.pi/4,torch.pi/4)],
            'A vehicle has a head-on collision with the ego vehicle': [(-torch.pi,-3*torch.pi/4),(3*torch.pi/4,torch.pi)],
            'A vehicle has a T-bone collision with the ego vehicle': [(torch.pi/4,3*torch.pi/4), (-torch.pi/4,-3*torch.pi/4)]
        }
    
    def transform_to_i_frame(self, pos_pred_i_world, pos_pred_j_world, yaw_pred_i_world, yaw_pred_j_world):
        """
        Transform j's positions and yaws from the world coordinate system to i's local coordinate system using PyTorch.

        Args:
            pos_pred_i_world: Tensor of shape (N, T, 2), positions of i in the world frame.
            pos_pred_j_world: Tensor of shape (N, T, 2), positions of j in the world frame.
            yaw_pred_i_world: Tensor of shape (N, T, 1), yaws of i in the world frame.
            yaw_pred_j_world: Tensor of shape (N, T, 1), yaws of j in the world frame.

        Returns:
            pos_j_in_i: Tensor of shape (N, T, 2), positions of j in i's local frame.
            yaw_j_in_i: Tensor of shape (N, T, 1), yaws of j in i's local frame.
        """
        # Compute relative positions: pos_j_world - pos_i_world
        relative_positions = pos_pred_j_world - pos_pred_i_world  # Shape (N, T, 2)
        
        # Compute rotation matrix for each yaw_i
        cos_yaw = torch.cos(-yaw_pred_i_world.squeeze(-1))  # Shape (N, T)
        sin_yaw = torch.sin(-yaw_pred_i_world.squeeze(-1))  # Shape (N, T)
        
        rotation_matrix = torch.stack([
            torch.stack([cos_yaw, -sin_yaw], dim=-1),  # First row of rotation matrix
            torch.stack([sin_yaw,  cos_yaw], dim=-1)   # Second row of rotation matrix
        ], dim=-2)  # Shape (N, T, 2, 2)
        
        # Rotate relative positions to i's local frame
        pos_j_in_i = torch.einsum("ntij,ntj->nti", rotation_matrix, relative_positions)  # Shape (N, T, 2)
        
        # Compute relative yaw: yaw_j_world - yaw_i_world,-
        yaw_j_in_i = yaw_pred_j_world - yaw_pred_i_world  # Shape (N, T, 1)
        yaw_j_in_i = (yaw_j_in_i + torch.pi) % (2 * torch.pi) - torch.pi 
        
        return pos_j_in_i, yaw_j_in_i
    def get_bounding_box_corners(self, centroid: torch.Tensor, yaw: torch.Tensor, extent: torch.Tensor) -> torch.Tensor:
        """(4, 2)"""
        x, y = centroid[0], centroid[1]
        sin, cos = torch.sin(yaw), torch.cos(yaw)
        width, length = extent[0] / 2, extent[1] / 2

        corners = torch.zeros((4, 2), device=centroid.device, dtype=centroid.dtype)
        # 
        corners[0, 0] = x + width * cos - length * sin
        corners[0, 1] = y + width * sin + length * cos
        # 
        corners[1, 0] = x - width * cos - length * sin
        corners[1, 1] = y - width * sin + length * cos
        # 
        corners[2, 0] = x - width * cos + length * sin
        corners[2, 1] = y - width * sin - length * cos
        # 
        corners[3, 0] = x + width * cos + length * sin
        corners[3, 1] = y + width * sin - length * cos
        
        return corners

    def get_sides_lengths(self, corners: torch.Tensor) -> torch.Tensor:
        """
        
        (4,)[]
        """
        # 
        front = torch.norm(corners[0] - corners[1], p=2)
        rear = torch.norm(corners[2] - corners[3], p=2)
        left = torch.norm(corners[1] - corners[2], p=2)
        right = torch.norm(corners[0] - corners[3], p=2)
        
        return torch.tensor([front, rear, left, right], device=corners.device, dtype=corners.dtype)

    def detect_collision(self, ego_pos: torch.Tensor, ego_yaw: torch.Tensor, ego_extent: torch.Tensor,
                        other_pos: torch.Tensor, other_yaw: torch.Tensor, other_extent: torch.Tensor):
        """
        
        (SAT)
        """
        # 
        ego_corners = self.get_bounding_box_corners(ego_pos, ego_yaw, ego_extent)
        other_corners = self.get_bounding_box_corners(other_pos, other_yaw, other_extent)
        
        # None
        if not self.check_collision_sat(ego_corners, other_corners):
            return None
        
        # 
        # 
        
        # 
        center_distance = torch.norm(ego_pos - other_pos, p=2)
        
        # 
        ego_sides = self.get_sides_lengths(ego_corners)
        other_sides = self.get_sides_lengths(other_corners)
        
        # 
        # 
        ego_diagonal = torch.sqrt(ego_extent[0]**2 + ego_extent[1]**2)
        other_diagonal = torch.sqrt(other_extent[0]**2 + other_extent[1]**2)
        
        # 
        penetration = torch.max(torch.zeros_like(center_distance), 
                            (ego_diagonal + other_diagonal) / 2 - center_distance)
        
        # 
        # 
        relative_yaw = (other_yaw - ego_yaw + torch.pi) % (2 * torch.pi) - torch.pi
        
        # 
        # 
        front_ratio = torch.abs(torch.cos(relative_yaw)) * 0.5 + 0.5
        rear_ratio = torch.abs(torch.cos(relative_yaw + torch.pi)) * 0.5 + 0.5
        left_ratio = torch.abs(torch.cos(relative_yaw + torch.pi/2)) * 0.5 + 0.5
        right_ratio = torch.abs(torch.cos(relative_yaw - torch.pi/2)) * 0.5 + 0.5
        
        # 
        front_length = penetration * front_ratio
        rear_length = penetration * rear_ratio
        left_length = penetration * left_ratio
        right_length = penetration * right_ratio
        
        return front_length, rear_length, left_length, right_length

    def check_collision_sat(self, corners1: torch.Tensor, corners2: torch.Tensor) -> bool:
        """
        (SAT)
        """
        for polygon in [corners1, corners2]:
            # 
            edges = torch.zeros((4, 2), device=corners1.device, dtype=corners1.dtype)
            for i in range(4):
                edges[i] = polygon[(i+1) % 4] - polygon[i]
            
            # 
            normals = torch.zeros((4, 2), device=corners1.device, dtype=corners1.dtype)
            for i in range(4):
                normals[i, 0] = -edges[i, 1]
                normals[i, 1] = edges[i, 0]
                # 
                normals[i] = normals[i] / torch.norm(normals[i], p=2)
            
            # 
            for normal in normals:
                # 
                min1, max1 = float('inf'), float('-inf')
                min2, max2 = float('inf'), float('-inf')
                
                for corner in corners1:
                    projection = torch.dot(corner, normal)
                    min1 = min(min1, projection.item())
                    max1 = max(max1, projection.item())
                
                for corner in corners2:
                    projection = torch.dot(corner, normal)
                    min2 = min(min2, projection.item())
                    max2 = max(max2, projection.item())
                
                # 
                if max1 < min2 or max2 < min1:
                    # 
                    return False
        
        # 
        return True

    def forward(self, x, data_batch, agt_mask=None):
        data_extent = data_batch["extent"]
        data_extent_i = data_extent[self.target_ind]
        data_extent_j = data_extent[self.ref_ind]
        
        B, N, T, _ = x.shape
        if agt_mask is not None:
            x = x[agt_mask]
        
        # 
        pos_pred = x[..., :2]
        yaw_pred = x[..., 3:4]
        
        # 
        pos_pred_world, yaw_pred_world = transform_coord_agents_to_world(pos_pred, yaw_pred, data_batch)
        
        # 
        pos_pred_i_world = select_agent_ind(pos_pred_world, self.target_ind).detach().clone()
        pos_pred_j_world = select_agent_ind(pos_pred_world, self.ref_ind)
        
        yaw_pred_i_world = select_agent_ind(yaw_pred_world, self.target_ind).detach().clone()
        yaw_pred_j_world = select_agent_ind(yaw_pred_world, self.ref_ind)
        
        # 
        loss = torch.zeros(pos_pred_i_world.shape[0], device=pos_pred_i_world.device)
        
        # 
        for n in range(pos_pred_i_world.shape[0]):
            for t in range(pos_pred_i_world.shape[1]):
                coll = self.detect_collision(
                    ego_pos=pos_pred_i_world[n, t],
                    ego_yaw=yaw_pred_i_world[n, t],
                    ego_extent=data_extent_i,
                    other_pos=pos_pred_j_world[n, t],
                    other_yaw=yaw_pred_j_world[n, t],
                    other_extent=data_extent_j,
                )
                
                if coll is not None:
                    front_length, rear_length, left_length, right_length = coll
                    curr_yaw_i = yaw_pred_i_world[n, t]
                    curr_yaw_j = yaw_pred_j_world[n, t]
                    relative_yaw = (curr_yaw_j - curr_yaw_i + torch.pi) % (2 * torch.pi) - torch.pi
                    
                    # 
                    yaw_list_points = []
                    right_yaw = False
                    for yaw_list in self.yaw_collision_type_dict[self.collision_type]:
                        down, up = yaw_list
                        yaw_list_points.append(down)
                        yaw_list_points.append(up)
                        if down <= relative_yaw <= up:
                            yaw_loss = torch.tensor(0.0, device=pos_pred_i_world.device)
                            right_yaw = True
                    
                    if not right_yaw:
                        yaw_differences = torch.tensor([torch.abs(relative_yaw - yaw) for yaw in yaw_list_points], 
                                                    device=pos_pred_i_world.device)
                        yaw_loss = torch.min(yaw_differences)
                    
                    # 
                    if self.collision_type == 'A vehicle cuts in and collides with the ego vehicle':
                        length_loss = torch.clamp(torch.max(front_length, rear_length) - torch.max(left_length, right_length), min=0)
                    elif self.collision_type == 'A vehicle collides with the ego vehicle':
                        length_loss = torch.clamp(torch.max(torch.max(front_length, left_length), right_length) - rear_length, min=0)
                    elif self.collision_type == 'Ego vehicle rear-ends another vehicle':
                        length_loss = torch.clamp(torch.max(torch.max(rear_length, left_length), right_length) - front_length, min=0)
                    elif self.collision_type == 'A vehicle has a head-on collision with the ego vehicle':
                        length_loss = torch.clamp(torch.max(torch.max(rear_length, left_length), right_length) - front_length, min=0)
                    elif self.collision_type == 'A vehicle has a T-bone collision with the ego vehicle':
                        length_loss = torch.clamp(torch.max(front_length, rear_length) - torch.max(left_length, right_length), min=0)
                    else:
                        raise NotImplementedError
                    
                    loss[n] = yaw_loss + length_loss
        
        return loss
    # def get_bounding_box(self,centroid: torch.Tensor, yaw: torch.Tensor, extent: torch.Tensor) -> Polygon:
    #     x, y = centroid[0], centroid[1]
    #     sin, cos = torch.sin(yaw), torch.cos(yaw)
    #     width, length = extent[0] / 2, extent[1] / 2

    #     x1, y1 = (x + width * cos - length * sin, y + width * sin + length * cos)
    #     x2, y2 = (x + width * cos + length * sin, y + width * sin - length * cos)
    #     x3, y3 = (x - width * cos + length * sin, y - width * sin - length * cos)
    #     x4, y4 = (x - width * cos - length * sin, y - width * sin + length * cos)
    #     return Polygon([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    # def get_sides(self,bbox: Polygon) -> Tuple[LineString, LineString, LineString, LineString]:
    #     (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bbox.exterior.coords[:-1]
    #     return (
    #         LineString([(x1, y1), (x2, y2)]),
    #         LineString([(x3, y3), (x4, y4)]),
    #         LineString([(x1, y1), (x4, y4)]),
    #         LineString([(x2, y2), (x3, y3)]),
    #     )

    # def detect_collision(
    #     self,
    #     ego_pos: torch.Tensor,
    #     ego_yaw: torch.Tensor,
    #     ego_extent: torch.Tensor,
    #     other_pos: torch.Tensor,
    #     other_yaw: torch.Tensor,
    #     other_extent: torch.Tensor,
    # ):
    #     ego_bbox = self.get_bounding_box(centroid=ego_pos, yaw=ego_yaw, extent=ego_extent)
    #     import warnings
    #     warnings.filterwarnings('ignore')
        
    #     agent_box=self.get_bounding_box(centroid=other_pos, yaw=other_yaw, extent=other_extent)
    #     if ego_bbox.intersects(agent_box):
    #         front_side, rear_side, left_side, right_side = self.get_sides(agent_box)
    #         front_int = agent_box.intersection(front_side)
    #         rear_int = agent_box.intersection(rear_side)
    #         left_int = agent_box.intersection(left_side)
    #         right_int = agent_box.intersection(right_side)
    #         return front_int.length, rear_int.length, left_int.length, right_int.length
    #     return None
    # def forward(self, x, data_batch, agt_mask=None):
    #     data_extent = data_batch["extent"]
    #     data_extent_i = data_extent[self.target_ind]
    #     data_extent_j = data_extent[self.ref_ind]
    #     length_i = data_extent_i[0]
    #     width_i = data_extent_i[1]
    #     length_j = data_extent_j[0]
    #     width_j = data_extent_j[1]
    #     # self.collision_radius = length_i/2+length_j/2+self.collision_radius

    #     B, N, T, _ = x.shape
    #     if agt_mask is not None:
    #         x = x[agt_mask]
    #     # ego_vehicleyaw
    #     # Select positions
    #     # (B,N,T,6) -> (B,N,T,2)
    #     pos_pred = x[..., :2]
    #     # Select yaws
    #     # (B,N,T,6) -> (B,N,T,1)
    #     yaw_pred = x[..., 3:4]
    #     # convert prediction from the respective agent coordinates to the world coordinate
    #     # (B,N,T,2), (B,N,T,1), dict -> (B,N,T,2), (B,N,T,1)
    #     pos_pred_world, yaw_pred_world = transform_coord_agents_to_world(pos_pred, yaw_pred, data_batch)

    #     # select the relevant agents with index self.target_ind in the world coordinate
    #     # (B, N, T, 2), int -> (N, T, 2)
    #     pos_pred_i_world = select_agent_ind(pos_pred_world, self.target_ind).detach().clone()
    #     # select the relevant agents with index self.ref_ind in the world coordinate
    #     # (B, N, T, 2), int -> (N, T, 2)
    #     pos_pred_j_world = select_agent_ind(pos_pred_world, self.ref_ind)
        
    #     yaw_pred_i_world = select_agent_ind(yaw_pred_world, self.target_ind).detach().clone()
    #     yaw_pred_j_world = select_agent_ind(yaw_pred_world, self.ref_ind)
    #     loss=torch.zeros(pos_pred_i_world.shape[0])
    #     for n in range(pos_pred_i_world.shape[0]):
    #         for t in range(pos_pred_i_world.shape[1]):
    #             coll = self.detect_collision(
    #                 ego_pos=pos_pred_i_world[n, t],
    #                 ego_yaw=yaw_pred_i_world[n, t],
    #                 ego_extent=data_extent_i,
    #                 other_pos=pos_pred_j_world[n, t],
    #                 other_yaw=yaw_pred_j_world[n, t],
    #                 other_extent=data_extent_j,
    #             )
    #             if coll is not None:
    #                 front_length, rear_length, left_length, right_length = coll
    #                 curr_yaw_i = yaw_pred_i_world[n, t]
    #                 curr_yaw_j = yaw_pred_j_world[n, t]
    #                 relative_yaw = (curr_yaw_j - curr_yaw_i+ torch.pi) % (2 * torch.pi) - torch.pi 
    #                 yaw_list_points = []
    #                 right_yaw=False
    #                 for yaw_list in self.yaw_collision_type_dict[self.collision_type]:
    #                     down, up=yaw_list
    #                     yaw_list_points.append(down)
    #                     yaw_list_points.append(up)
    #                     if down <= relative_yaw <= up:
    #                         yaw_loss = torch.tensor(0)
    #                         right_yaw=True
    #                 if not right_yaw:
    #                     yaw_loss = torch.min(torch.tensor([torch.abs(relative_yaw - yaw) for yaw in yaw_list_points]))

    #                 if self.collision_type == 'A vehicle cuts in and collides with the ego vehicle':
    #                     length_loss= torch.clip(torch.max(front_length, rear_length)-torch.max(left_length, right_length), min=0)
    #                 elif self.collision_type == 'A vehicle collides with the ego vehicle':
    #                     length_loss = torch.clip(torch.max(front_length, left_length, right_length)-rear_length, min=0)
    #                 elif self.collision_type == 'Ego vehicle rear-ends another vehicle':
    #                     length_loss = torch.clip(torch.max(rear_length, left_length, right_length)-front_length, min=0)
    #                 elif self.collision_type == 'A vehicle has a head-on collision with the ego vehicle':
    #                     length_loss = torch.clip(torch.max(rear_length, left_length, right_length)-front_length, min=0)
    #                 elif self.collision_type == 'A vehicle has a T-bone collision with the ego vehicle':
    #                     length_loss= torch.clip(torch.max(front_length, rear_length)-torch.max(left_length, right_length), min=0)
    #                 else:
    #                     raise NotImplementedError
    #                 loss[n] = yaw_loss + length_loss
    #     return loss
        #mask
        mask = torch.ones_like(yaw_pred_i_world[:,:,0])
        #collision_radius0
        mask = torch.where(torch.norm(pos_pred_i_world - pos_pred_j_world, dim=-1) > self.collision_radius, torch.zeros_like(mask), mask)
        #ijyaw
        # (N, T, 2), (N, T, 2) -> (N, T, 2)
        pos_j_in_i, yaw_j_in_i = self.transform_to_i_frame(pos_pred_i_world, pos_pred_j_world, yaw_pred_i_world, yaw_pred_j_world)
        #yawloss
        yaw_target = self.yaw_collision_type_dict[self.collision_type]
        yaw_loss_list = []
        for yaw in yaw_target:
            yaw_loss_list.append(torch.abs(torch.abs(yaw_j_in_i - yaw)))
        #
        yaw_loss,_ = torch.min(torch.cat(yaw_loss_list, dim=2), dim=-1, keepdim=True)
        #mask
        yaw_loss = yaw_loss * (mask.unsqueeze(-1))
        #xloss
        if self.collision_type == 'A vehicle cuts in and collides with the ego vehicle':
            #0jlength+ilengthloss
            # if pos_j_in_i[..., 1] < 0:
            #     y_loss = torch.abs(pos_j_in_i[..., 1])
            # elif pos_j_in_i[..., 1] > length_i/2 + length_/2:
            #     y_loss = torch.abs(pos_j_in_i[..., 1]-length_i/2-length_j/2) 
            # else:
            #     y_loss = torch.zeros_like(pos_j_in_i[..., 1])
            mask_pos_j_in_i_small_than_0=pos_j_in_i[..., 1] < 0
            mask_pos_j_in_i_big_than_length=pos_j_in_i[..., 1] > (length_i/2 + length_j/2)
            y_loss = torch.where(mask_pos_j_in_i_small_than_0, torch.abs(pos_j_in_i[..., 1]), torch.zeros_like(pos_j_in_i[..., 1]))
            y_loss = torch.where(mask_pos_j_in_i_big_than_length, torch.abs(pos_j_in_i[..., 1]-length_i/2-length_j/2), y_loss)
        elif self.collision_type == 'A vehicle rear-ends the ego vehicle':
            y_loss = torch.abs(pos_j_in_i[..., 1]+length_i/2+length_j/2) 
        elif self.collision_type == 'Ego vehicle rear-ends another vehicle':
            y_loss = torch.abs(pos_j_in_i[..., 1]-length_i/2-length_j/2) 
        elif self.collision_type == 'A vehicle has a head-on collision with the ego vehicle':
            y_loss = torch.abs(pos_j_in_i[..., 1]-length_i/2-length_j/2) 
        elif self.collision_type == 'A vehicle has a T-bone collision with the ego vehicle':
            y_loss = torch.abs(pos_j_in_i[..., 1]) 
        else:
            raise NotImplementedError
        y_loss=(y_loss.unsqueeze(-1))*(mask.unsqueeze(-1))
        #loss
        return (yaw_loss + y_loss)[...,0].mean(-1)
        
# TODO target waypoint guidance
#       - Really the target positions should be global not local, will have to do some extra work to transform into
#           the local frame.
class TargetPosAtTimeLoss(GuidanceLoss):
    '''
    Hit a specific waypoint at a specific time step (within the current planning horizon).
    '''
    def __init__(self, target_pos, target_time):
        '''
        - target_pos : (B,2) batch of positions to hit, B must equal the number of agents after applying mask in forward.
        - target_time: (B,) batch of times at which to hit the given positions
        '''
        super().__init__()
        self.set_target(target_pos, target_time)

    def set_target(self, target_pos, target_time):
        if isinstance(target_pos, torch.Tensor):
            self.target_pos = target_pos
        else:
            self.target_pos = torch.tensor(target_pos)
        if isinstance(target_time, torch.Tensor):
            self.target_time = target_time
        else:
            self.target_time = torch.tensor(target_time)

    def forward(self, x, data_batch, agt_mask=None):
        '''
        - x : the current trajectory (B, N, T, 6) where N is the number of samples and 6 is (x, y, vel, yaw, acc, yawvel)
        '''
        if agt_mask is not None:
            x = x[agt_mask]
        assert x.size(0) == self.target_pos.size(0)
        assert x.size(0) == self.target_time.size(0)
        
        x_pos = x[torch.arange(x.size(0)), :, self.target_time, :2]
        tgt_pos = self.target_pos.to(x_pos.device)[:,None] # (B,1,2)
        # MSE
        # loss = torch.sum((x_pos - tgt_pos)**2, dim=-1)
        loss = torch.norm(x_pos - tgt_pos, dim=-1)
        # # Normalization Change: clip to 1
        # loss = torch.clip(loss, max=1)
        return loss

class TargetPosLoss(GuidanceLoss):
    '''
    Hit a specific waypoint at some time step (within the current planning horizon).
    '''
    def __init__(self, target_pos, min_target_time=0.0):
        '''
        - target_pos : (B,2) batch of positions to hit, B must equal the number of agents after applying mask in forward.
        - min_target_time : float, only tries to hit the target after the initial min_target_time*horizon_num_steps of the trajectory
                            e.g. if = 0.5 then only the last half of the trajectory will attempt to go through target
        '''
        super().__init__()
        self.min_target_time = min_target_time
        self.set_target(target_pos)

    def set_target(self, target_pos):
        if isinstance(target_pos, torch.Tensor):
            self.target_pos = target_pos
        else:
            self.target_pos = torch.tensor(target_pos)


    def forward(self, x, data_batch, agt_mask=None):
        '''
        - x : the current trajectory (B, N, T, 6) where N is the number of samples and 6 is (x, y, vel, yaw, acc, yawvel)
        '''
        if agt_mask is not None:
            x = x[agt_mask]
        assert x.size(0) == self.target_pos.size(0)
        
        min_t = int(self.min_target_time*x.size(2))
        x_pos = x[:,:,min_t:,:2]
        tgt_pos = self.target_pos.to(x_pos.device)[:,None,None] # (B,1,1,2)
        dist = torch.norm(x_pos - tgt_pos, dim=-1)
        # give higher loss weight to the closest valid timesteps
        loss_weighting = F.softmin(dist, dim=-1)
        loss = loss_weighting * torch.sum((x_pos - tgt_pos)**2, dim=-1) # (B, N, T)
        # loss = loss_weighting * torch.norm(x_pos - tgt_pos, dim=-1)
        loss = torch.mean(loss, dim=-1) # (B, N)
        # # Normalization Change: clip to 1
        # loss = torch.clip(loss, max=1)
        return loss

# TODO: this currently depends on the map that's also passed into the network.
#       if the network map viewport is small and the future horizon is long enough,
#       it may go outside the range of the map and then this is really inaccurate.
class MapCollisionLoss(GuidanceLoss):
    '''
    Agents should not go offroad.
    '''
    def __init__(self, num_points_lw=(10, 10), decay_rate=0.9, guide_moving_speed_th=5e-1):
        '''
        - num_points_lw : how many points will be sampled within each agent bounding box
                            to detect map collisions. e.g. (15, 10) will sample a 15 x 10 grid
                            of points where 15 is along the length and 10 along the width.
        '''
        super().__init__()
        self.num_points_lw = num_points_lw
        self.decay_rate = decay_rate
        self.guide_moving_speed_th = guide_moving_speed_th
        lwise = torch.linspace(-0.5, 0.5, self.num_points_lw[0])
        wwise = torch.linspace(-0.5, 0.5, self.num_points_lw[1])
        self.local_coords = torch.cartesian_prod(lwise, wwise)#,
        # TODO could cache initial (local) point samplings if given extents at instantiation

    def gen_agt_coords(self, pos, yaw, lw, raster_from_agent):
        '''
        - pos : B x 2
        - yaw : B x 1
        - lw : B x 2
        '''
        B = pos.size(0)
        # self.local_coordspos0B
        cur_loc_coords = self.local_coords.to(pos.device).unsqueeze(0).expand((B, -1, -1))#B,100,2
        # scale by the extents
        cur_loc_coords = cur_loc_coords * lw.unsqueeze(-2)

        # transform initial coords to given pos, yaw
        s = torch.sin(yaw).unsqueeze(-1)
        c = torch.cos(yaw).unsqueeze(-1)
        rotM = torch.cat((torch.cat((c, s), dim=-1), torch.cat((-s, c), dim=-1)), dim=-2)
        agt_coords_agent_frame = cur_loc_coords @ rotM + pos.unsqueeze(-2)
        
        # then transform to raster frame
        agt_coords_raster_frame = GeoUtils.transform_points_tensor(agt_coords_agent_frame, raster_from_agent)

        # # NOTE: debug viz sanity check
        # import matplotlib
        # import matplotlib.pyplot as plt
        # agt_coords = agt_coords.reshape((8, 10, 52, 25, 2))
        # fig = plt.figure()
        # for t in range(agt_coords.size(2)):
        #     plt.scatter(agt_coords[3,0,t,:,0].cpu().detach().numpy(),
        #                 agt_coords[3,0,t,:,1].cpu().detach().numpy())
        # # plt.gca().set_xlim([-1, 7])
        # # plt.gca().set_ylim([-4, 4])
        # plt.axis('equal')
        # plt.show()
        # plt.close(fig)

        return agt_coords_agent_frame, agt_coords_raster_frame

    def forward(self, x, data_batch, agt_mask=None):   
        drivable_map = data_batch["drivable_map"] #map
        data_extent = data_batch["extent"]
        data_raster_from_agent = data_batch["raster_from_agent"]#agent
        cur_speed= data_batch['curr_speed']
        if agt_mask is not None:
            x = x[agt_mask]
            drivable_map = drivable_map[agt_mask]
            data_extent = data_extent[agt_mask]
            data_raster_from_agent = data_raster_from_agent[agt_mask]
            cur_speed = cur_speed[agt_mask]
        _, H, W = drivable_map.size()

        B, N, T, _ = x.size()
        traj = x.reshape((-1, 6)) # B*N*T x 6
        pos_pred = traj[:,:2]
        yaw_pred = traj[:, 3:4] 
        lw = data_extent[:,None,None].expand((B, N, T, 3)).reshape((-1, 3))[:,:2]
        diag_len = torch.sqrt(torch.sum(lw*lw, dim=-1))# diagonal length 
        data_raster_from_agent = data_raster_from_agent[:,None,None].expand((B, N, T, 3, 3)).reshape((-1, 3, 3))

        # sample points within each agent to check if drivable
        agt_samp_pts, agt_samp_pix = self.gen_agt_coords(pos_pred, yaw_pred, lw, data_raster_from_agent)
        # agt_samp_pts = agt_samp_pts.reshape((B, N, T, -1, 2))
        agt_samp_pix = agt_samp_pix.reshape((B, N, T, -1, 2)).long().detach() # only used to query drivable map, not to compute loss
        # NOTE: this projects pixels outside the map onto the edge
        agt_samp_l = torch.clamp(agt_samp_pix[..., 0:1], 0, W-1)
        agt_samp_w = torch.clamp(agt_samp_pix[..., 1:2], 0, H-1)
        agt_samp_pix = torch.cat([agt_samp_l, agt_samp_w], dim=-1)

        # query these points in the drivable area to determine collision
        _, P, _ = agt_samp_pts.size()
        map_coll_mask = torch.isclose(batch_detect_off_road(agt_samp_pix, drivable_map), torch.ones((1)).to(agt_samp_pix.device))
        map_coll_mask = map_coll_mask.reshape((-1, P)) #1

        # only apply loss to timesteps that are partially overlapping
        per_step_coll = torch.sum(map_coll_mask, dim=-1)
        overlap_mask = ~torch.logical_or(per_step_coll == 0, per_step_coll == P)#

        overlap_coll_mask = map_coll_mask[overlap_mask]
        overlap_agt_samp = agt_samp_pts[overlap_mask]
        overlap_diag_len = diag_len[overlap_mask]

        #
        # The idea here: for each point that is offroad, we want to compute
        #   the minimum distance to a point that is on the road to give a nice
        #   gradient to push it back.
        #

        # compute dist mat between all pairs of points at each step
        # NOTE: the detach here is a very subtle but IMPORTANT point
        #       since these sample points are a function of the pos/yaw, if we compute
        #       the distance between them the gradients will always be 0, no matter how
        #       we change the pos and yaw the distance will never change. But if we detach
        #       one and compute distance to these arbitrary points we've selected, then
        #       we get a useful gradient.
        #           Moreover, it's also importan the columns are the ones detached here!
        #       these correspond to the points that ARE colliding. So if we try to max
        #       distance b/w these and the points inside the agent, it will push the agent
        #       out of the offroad area. If we do it the other way it will pull the agent
        #       into the offroad (if we max the dist) or just be a small pull in the correct dir
        #       (if we min the dist).
        pt_samp_dist = torch.cdist(overlap_agt_samp, overlap_agt_samp.clone().detach())
        # get min dist just for points still on the road
        # so we mask out points off the road (this also removes diagonal for off-road points which excludes self distances)
        pt_samp_dist = torch.where(overlap_coll_mask.unsqueeze(-1).expand(-1, -1, P),
                                   torch.ones_like(pt_samp_dist)*np.inf,
                                   pt_samp_dist)#,
        pt_samp_min_dist_all = torch.amin(pt_samp_dist, dim=1) # previously masked rows, so compute min over cols
        # compute actual loss
        pt_samp_loss_all = 1.0 - (pt_samp_min_dist_all / overlap_diag_len.unsqueeze(1))
        # only want a loss for off-road points
        pt_samp_loss_offroad = torch.where(overlap_coll_mask,
                                               pt_samp_loss_all,
                                               torch.zeros_like(pt_samp_loss_all))

        overlap_coll_loss = torch.sum(pt_samp_loss_offroad, dim=-1)
        # expand back to all steps, other non-overlap steps will be zero
        all_coll_loss = torch.zeros((agt_samp_pts.size(0))).to(overlap_coll_loss.device)
        all_coll_loss[overlap_mask] = overlap_coll_loss
        
        # summing over timesteps
        # all_coll_loss = all_coll_loss.reshape((B, N, T)).sum(-1)

        # consider offroad only for those moving vehicles
        all_coll_loss = all_coll_loss.reshape((B, N, T))
        moving = torch.abs(cur_speed) > self.guide_moving_speed_th
        moving_mask = moving.view((B,1,1)).expand(B, N, T)
        all_coll_loss = torch.where(moving_mask,
                                    all_coll_loss,
                                    torch.zeros_like(all_coll_loss))

        # penalize early steps more than later steps
        exp_weights = torch.tensor([self.decay_rate ** t for t in range(T)], device=all_coll_loss.device)
        exp_weights /= exp_weights.sum()
        all_coll_loss = all_coll_loss * exp_weights[None, None, :]
        all_coll_loss = all_coll_loss.sum(-1)

        return all_coll_loss

#
# Global waypoint target losses
#   (i.e. at some future planning horizon)
#
def compute_progress_loss(pos_pred, tgt_pos, urgency,
                          tgt_time=None,
                          pref_speed=1.42,
                          dt=0.1,
                          min_progress_dist=0.5):
    '''
    Evaluate progress towards a goal that we want to hit.
    - pos_pred : (B x N x T x 2)
    - tgt_pos : (B x 2)
    - urgency : (B) in (0.0, 1.0]
    - tgt_time : [optional] (B) local target time, i.e. starting from the current t0 how many steps in the
                    future will we need to hit the target. If given, loss is computed to cover the distance
                    necessary to hit the goal at the given time
    - pref_speed: speed used to determine how much distance should be covered in a time interval
    - dt : step interval of the trajectories
    - min_progress_dist : float (in meters). if not using tgt_time, the minimum amount of progress that should be made in
                            each step no matter what the urgency is
    '''
    # TODO: use velocity or heading to avoid degenerate case of trying to whip around immediately
    #       and getting stuck from unicycle dynamics?

    # distance from final trajectory timestep to the goal position
    final_dist = torch.norm(pos_pred[:,:,-1] - tgt_pos[:,None], dim=-1)

    if tgt_time is not None:
        #
        # have a target time: distance covered is based on arrival time
        #
        # distance of straight path from current pos to goal at the average speed

        goal_dist = tgt_time * dt * pref_speed

        # factor in urgency (shortens goal_dist since we can't expect to always go on a straight path)
        goal_dist = goal_dist * (1.0 - urgency)
        # only apply loss if above the goal distance
        progress_loss = F.relu(final_dist - goal_dist[:,None])
    else:
        #
        # don't have a target time: distance covered based on making progress
        #       towards goal with the specified urgency
        #
        # following straight line path from current pos to goal
        max_horizon_dist = pos_pred.size(2) * dt * pref_speed
        # at max urgency, want to cover distance of this straight line path
        # at min urgency, just make minimum progress
        goal_dist = torch.maximum(urgency * max_horizon_dist, torch.tensor([min_progress_dist]).to(urgency.device))

        init_dist = torch.norm(pos_pred[:,:,0] - tgt_pos[:,None], dim=-1)
        progress_dist = init_dist - final_dist
        # only apply loss if less progress than goal
        progress_loss = F.relu(goal_dist[:,None] - progress_dist)

    return progress_loss

class GlobalTargetPosAtTimeLoss(GuidanceLoss):
    '''
    Hit a specific waypoint at a specific time step (in some future planning horizon).
    '''
    def __init__(self, target_pos, target_time, urgency, pref_speed=1.42, dt=0.1, target_tolerance=2, action_num=5):
        '''
        - target_pos : (B,2) batch of GLOBAL positions to hit, B must equal the number of agents after applying mask in forward.
        - target_time: (B,) batch of GLOBAL times at which to hit the given positions
        - urgency: (B,) batch of [0.0, 1.0] urgency factors for each agent
                        The larger the urgency, the closer the agent will try to
                        to be at each planning step. This is used to scale the goal distance, i.e.
                        with urgency of 0.0, the agent will try to be close enough to the target
                        that they can take a straight path and get there on time. With urgency 1.0,
                        the agent will try to already be at the goal at the last step of every planning step.
        - pref_speed: (B,), speed used to determine how much distance should be covered in a time interval.
        - dt : of the timesteps that will be passed in (i.e. the diffuser model)

        - target_tolerance : if not None, then within this tolerance distance, the loss will be masked.
        - action_num: how many actions per rollout step.
        '''
        super().__init__()
        self.set_target(target_pos, target_time)
        self.urgency = torch.tensor(urgency)
        self.pref_speed = torch.tensor(pref_speed, dtype=torch.float32)
        self.dt = dt
        # create local loss to use later when within reach
        #       will update target_pos/time later as necessary
        self.local_tgt_loss = TargetPosAtTimeLoss(target_pos, target_time)

        self.target_tolerance = target_tolerance
        self.action_num = action_num
        self.have_reached_mask = None

    def set_target(self, target_pos, target_time):
        self.target_pos = torch.tensor(target_pos)
        self.target_time = torch.tensor(target_time)

    def forward(self, x, data_batch, agt_mask=None):
        '''
        - x : the current trajectory (B, N, T, 6) where N is the number of samples and 6 is (x, y, vel, yaw, acc, yawvel)
        '''
        agent_from_world = data_batch["agent_from_world"]
        world_from_agent = data_batch["world_from_agent"]
        agent_hist = data_batch['agent_hist']
        if agt_mask is not None:
            x = x[agt_mask]
            agent_from_world = agent_from_world[agt_mask]
            world_from_agent = world_from_agent[agt_mask]
            agent_hist = agent_hist[agt_mask]
        assert x.size(0) == self.target_pos.size(0)
        assert x.size(0) == self.target_time.size(0)
        assert x.size(0) == self.urgency.size(0)
        assert x.size(0) == self.pref_speed.size(0)

        if self.have_reached_mask is None:
            self.have_reached_mask = torch.zeros((x.size(0), x.size(1)), dtype=torch.bool).to(x.device)

        # transform world targets to agent frame
        local_target_pos = GeoUtils.transform_points_tensor(self.target_pos[:,None].to(x.device), agent_from_world)[:,0]

        # decide which agents need progress loss vs. exact target loss
        local_target_time = self.target_time.to(x.device) - self.global_t
        horizon_len = x.size(2)
        # if within planning horizon but hasn't been passed yet
        exact_mask = torch.logical_and(local_target_time < horizon_len, local_target_time >= 0)
        # apply progress loss if not within planning horizon yet and hasn't been passed
        prog_mask = torch.logical_and(~exact_mask, local_target_time >= 0)

        loss = torch.zeros((x.size(0), x.size(1))).to(x)
        # progress loss
        num_exact = torch.sum(exact_mask)
        if num_exact != x.size(0):
            pos_pred = x[..., :2]
            progress_loss = compute_progress_loss(pos_pred[prog_mask],
                                                  local_target_pos[prog_mask],
                                                  self.urgency[prog_mask].to(x.device),
                                                  local_target_time[prog_mask],
                                                  self.pref_speed[prog_mask].to(x.device),
                                                  self.dt)
            loss[prog_mask] = progress_loss
        # exact target loss
        if  num_exact > 0:
            exact_local_tgt_pos = local_target_pos[exact_mask]
            exact_local_tgt_time = local_target_time[exact_mask]
            self.local_tgt_loss.set_target(exact_local_tgt_pos, exact_local_tgt_time)
            exact_loss = self.local_tgt_loss(x[exact_mask], None, None) # shouldn't need data_batch or agt_mask
            loss[exact_mask] = exact_loss


        if self.target_tolerance is not None:
            pos_hist = agent_hist[...,-self.action_num:,:2]
            pos_hist_world = GeoUtils.transform_points_tensor(pos_hist, world_from_agent)[:,0]

            dist_to_target = torch.norm(pos_hist_world - self.target_pos[:,None].to(x.device), dim=-1)
            min_dist_to_target = torch.min(dist_to_target, dim=-1)[0]
            
            self.have_reached_mask[min_dist_to_target < self.target_tolerance] = True

            # print('self.have_reached_mask', self.have_reached_mask)
            loss[self.have_reached_mask] = 0.0

        return loss

class GlobalTargetPosLoss(GuidanceLoss):
    '''
    Hit a specific waypoint at some time in the future.
    '''
    def __init__(self, target_pos, urgency, pref_speed=1.42, dt=0.1, min_progress_dist=0.5, target_tolerance=None, action_num=5):
        '''
        - target_pos : (B,2) batch of GLOBAL positions to hit, B must equal the number of agents after applying mask in forward.
        - urgency: (B,) batch of [0.0, 1.0] urgency factors for each agent
                        urgency in this case means how much of the maximal possible distance should
                        be covered in a single planning horizon. If urgency is 1.0 the agent
                        will shoot for a straight line path to the target. If it is 0.0 it will just
                        try to make the minimal amount of progress at each plan.
        - pref_speed: (B,), speed used to determine how much distance should be covered in a time interval.
        - dt : of the timesteps that will be passed in (i.e. the diffuser model)
        - min_progress_dist : minimum distance that should be covered in each plan no matter what the urgency is
        
        - target_tolerance : if not None, then within this tolerance distance, the loss will be masked.
        - action_num: how many actions per rollout step.
        '''
        super().__init__()
        self.set_target(target_pos)
        self.urgency = torch.tensor(urgency)
        self.pref_speed = torch.tensor(pref_speed, dtype=torch.float32)
        self.dt = dt
        self.min_progress_dist = min_progress_dist
        # create local loss to use later when within reach
        #       will update target_pos/time later as necessary
        self.local_tgt_loss = TargetPosLoss(target_pos)

        self.target_tolerance = target_tolerance
        self.action_num = action_num
        self.have_reached_mask = None

    def set_target(self, target_pos):
        self.target_pos = torch.tensor(target_pos)

    def forward(self, x, data_batch, agt_mask=None):
        '''
        - x : the current trajectory (B, N, T, 6) where N is the number of samples and 6 is (x, y, vel, yaw, acc, yawvel)
        '''
        agent_from_world = data_batch["agent_from_world"]
        world_from_agent = data_batch["world_from_agent"]
        agent_hist = data_batch['agent_hist']
        if agt_mask is not None:
            x = x[agt_mask]
            agent_from_world = agent_from_world[agt_mask]
            world_from_agent = world_from_agent[agt_mask]
            agent_hist = agent_hist[agt_mask]

        assert x.size(0) == self.target_pos.size(0)
        assert x.size(0) == self.urgency.size(0)
        assert x.size(0) == self.pref_speed.size(0)

        if self.have_reached_mask is None:
            self.have_reached_mask = torch.zeros((x.size(0), x.size(1)), dtype=torch.bool).to(x.device)

        # transform world targets to agent frame
        local_target_pos = GeoUtils.transform_points_tensor(self.target_pos[:,None].to(x.device), agent_from_world)[:,0]

        # decide which agents need progress loss vs. exact target loss
        # if agent can progress along straight line at preferred speed
        #       and arrive at target within the horizon, consider it in range
        horizon_len = x.size(2)
        single_horizon_dist = horizon_len * self.dt * self.pref_speed.to(x.device)
        local_target_dist = torch.norm(local_target_pos, dim=-1)
        exact_mask = local_target_dist < single_horizon_dist
        prog_mask = ~exact_mask

        loss = torch.zeros((x.size(0), x.size(1))).to(x)
        # progress loss
        num_exact = torch.sum(exact_mask)
        if num_exact != x.size(0):
            pos_pred = x[..., :2]
            progress_loss = compute_progress_loss(pos_pred[prog_mask],
                                                  local_target_pos[prog_mask],
                                                  self.urgency[prog_mask].to(x.device),
                                                  None,
                                                  self.pref_speed[prog_mask].to(x.device),
                                                  self.dt,
                                                  self.min_progress_dist)
            loss[prog_mask] = progress_loss
        # exact target loss
        if num_exact > 0:
            exact_local_tgt_pos = local_target_pos[exact_mask]
            self.local_tgt_loss.set_target(exact_local_tgt_pos)
            exact_loss = self.local_tgt_loss(x[exact_mask], None, None) # shouldn't need data_batch or agt_mask
            loss[exact_mask] = exact_loss
        

        if self.target_tolerance is not None:
            pos_hist = agent_hist[...,-self.action_num:,:2]
            pos_hist_world = GeoUtils.transform_points_tensor(pos_hist, world_from_agent)[:,0]

            dist_to_target = torch.norm(pos_hist_world - self.target_pos[:,None].to(x.device), dim=-1)
            min_dist_to_target = torch.min(dist_to_target, dim=-1)[0]

            # print('min_dist_to_target', min_dist_to_target)
            self.have_reached_mask[min_dist_to_target < self.target_tolerance] = True

            # print('self.have_reached_mask', self.have_reached_mask)
            loss[self.have_reached_mask] = 0.0

        return loss

class SocialGroupLoss(GuidanceLoss):
    '''
    Agents should move together.
    NOTE: this assumes full control over the scene. 
    '''
    def __init__(self, leader_idx=0, social_dist=1.5, cohesion=0.8):
        '''
        - leader_idx : index to serve as the leader of the group (others will follow them). This is the index in the scene, not the index within the specific social group.
        - social_dist : float, meters, How close members of the group will stand to each other.
        - cohesion : float [0.0, 1.0], at 1.0 essentially all group members try to be equidistant
                                            at 0.0 try to maintain distance only to closest neighbor and could get detached from rest of group
        '''
        super().__init__()
        self.leader_idx = leader_idx
        self.social_dist = social_dist
        assert cohesion >= 0.0 and cohesion <= 1.0
        self.random_neighbor_p = cohesion

    def forward(self, x, data_batch, agt_mask=None):
        data_world_from_agent = data_batch["world_from_agent"]
        pos_pred = x[..., :2]
        yaw_pred = x[..., 3:4]
        agt_idx = torch.arange(pos_pred.shape[0]).to(pos_pred.device)
        if agt_mask is not None:
            data_world_from_agent = data_world_from_agent[agt_mask]
            pos_pred = pos_pred[agt_mask]
            yaw_pred = yaw_pred[agt_mask]
            agt_idx = agt_idx[agt_mask]

        pos_pred_global, _ = transform_agents_to_world(pos_pred, yaw_pred, data_world_from_agent)

        # NOTE here we detach the leader pos, so that their motion is not affected by trying to stay close to the group
        #       this is so the group makes progress by following rather than just trying to be close together
        leader_mask = (agt_idx == self.leader_idx)[:,None,None,None].expand_as(pos_pred_global)
        pos_pred_global = torch.where(leader_mask, pos_pred_global.detach(), pos_pred_global)

        # print(leader_pos.size())
        # print(other_pos.size())

        # compute closest distance to others in social group
        B, N, T, _ = pos_pred_global.size()
        flat_pos_pred = pos_pred_global.transpose(0, 2).reshape((T*N, B, 2))
        flat_dist = torch.cdist(flat_pos_pred, flat_pos_pred) # T*N x B x B
        self_mask = torch.eye(B, device=flat_dist.device).unsqueeze(0).expand_as(flat_dist)
        flat_dist = torch.where(self_mask.bool(), np.inf*self_mask, flat_dist)  # mask out self-distances
        # pairs with neighbors based purely on min distance
        min_neighbor = torch.argmin(flat_dist, dim=-1)

        # randomly switch some closest neighbors to make more cohesive (but not to self)
        #       the idea is to avoid degenerate case where subset of agents create a connected component
        #       in nearest neighbor graph and drift from the rest of the group.
        # creates 2D matrix with self indices missing
        #   i.e for 4 agents [1, 2, 3]
        #                    [0, 2, 3]
        #                    [0, 1, 3]
        #                    [0, 2, 2]
        neighbor_choices = torch.arange(B)[None].expand((B,B)).masked_select(~torch.eye(B, dtype=bool)).view(B, B - 1).to(min_neighbor.device)
        neighbor_choices = neighbor_choices.unsqueeze(0).expand((T*N, B, B-1))
        # randomly sample one with p = self.random_neighbor_p
        rand_neighbor = torch.gather(neighbor_choices, 2, torch.randint(0, B-1, (T*N, B, 1)).to(neighbor_choices.device))[:,:,0]
        drop_mask = torch.rand((T*N, B)).to(min_neighbor.device) < self.random_neighbor_p
        neighbor_idx = torch.where(drop_mask,
                                   rand_neighbor,
                                   min_neighbor)

        # want assigned neighbor dist to be the desired social distance
        neighbor_dist = torch.gather(flat_dist, 2, neighbor_idx.unsqueeze(-1))[..., 0]
        neighbor_dist = neighbor_dist.reshape((T, N, B)).transpose(0, 2) # B, N, T
        loss = torch.mean((neighbor_dist - self.social_dist)**2, dim=-1)
        
        return loss

#-----------------------------------------------------------------------------------------------
# TBD: move to configs?
use_global_coord = True
use_until = False

class StopSignLoss(GuidanceLoss):
    '''
    A local stop-sign example.
    Vehicles are expected to have three consecutive time steps of low speed when passing the stop sign regions.
    '''
    def __init__(self, stop_sign_pos, stop_box_dim, scale, horizon_length, time_step_to_start, num_time_steps_to_stop, action_num, low_speed_th):
        '''
        - stop_sign_pos : stop sign region center position
        - stop_box_dim: stop box dimension (width, height)

        - scale: scale factor for smoothing out STL robust value (< 0 uses max/min, positive values closer to 0 gives more uniform gradients)

        - horizon_length: length of the horizon to use for computing STL robust value
        - time_step_to_start: time step to start using STL robust value
        - num_time_steps_to_stop: number of time steps to stop in the stop sign region
        
        - action_num: the number of action steps for each rollout step
        - low_speed_th: the speed threshold to use for the low speed condition
        '''
        super().__init__()
        self.stop_box_dim = torch.tensor(stop_box_dim)
        self.scale = torch.tensor(scale)

        self.horizon_length = horizon_length
        self.time_step_to_start = time_step_to_start
        self.num_time_steps_to_stop = num_time_steps_to_stop

        self.action_num = action_num
        self.low_speed_th = low_speed_th

        self.stop_sign = None
        self.stop_sign_pos = torch.tensor(stop_sign_pos)
        self.set_target(stop_sign_pos)
    
    def set_target(self, stop_sign_pos, N=None, device=None):
        '''
        invoked 1.at initialization 2.by GlobalStopSignLoss forward to reset stop sign pos at each rollout step
        '''
        if torch.is_tensor(stop_sign_pos):
            self.stop_sign_pos = stop_sign_pos.clone().detach()
        else:
            self.stop_sign_pos = torch.tensor(stop_sign_pos, device=device)

        if N is not None:
            if self.stop_sign is None:
                from tbsim.rules.stl_traffic_rules import StopSignRule

                # stop_sign_pos = self.stop_sign_pos.to(device)[:,None] # (B,1,2)
                # stop_sign_pos.expand(B, N, 2)
                # stop_sign_pos = TensorUtils.join_dimensions(stop_sign_pos, begin_axis=0, end_axis=2) # (B*N,2)

                stop_sign_pos = TensorUtils.repeat_by_expand_at(self.stop_sign_pos.to(device), N, dim=0)

                # stop_box_dim = self.stop_box_dim.to(device)[:,None] # (B,1,2)
                # stop_box_dim.expand(B, N, 2)
                # stop_box_dim = TensorUtils.join_dimensions(stop_box_dim, begin_axis=0, end_axis=2) # (B*N,2)

                stop_box_dim = TensorUtils.repeat_by_expand_at(self.stop_box_dim.to(device), N, dim=0)
        
                self.stop_sign = StopSignRule(stop_sign_pos, stop_box_dim, self.low_speed_th)
            else:
                # stop_sign_pos = self.stop_sign_pos[:,None] # (B,1,2)
                # stop_sign_pos.expand(B, N, 2)
                # stop_sign_pos = TensorUtils.join_dimensions(stop_sign_pos, begin_axis=0, end_axis=2) # (B*N, 2)
                stop_sign_pos = TensorUtils.repeat_by_expand_at(self.stop_sign_pos, N, dim=0)
                self.stop_sign.update_stop_box(stop_sign_pos=stop_sign_pos)

    def forward(self, x, data_batch, agt_mask=None, already_stopped=None):
        '''
        - x : the current trajectory (B, N, T, 6) where N is the number of samples and 6 is (x, y, vel, yaw, acc, yawvel)
        '''
        
        world_from_agent = data_batch["world_from_agent"]
        if agt_mask is not None:
            x = x[agt_mask]
            world_from_agent = world_from_agent[agt_mask]
        assert x.size(0) == self.stop_sign_pos.size(0) == self.stop_box_dim.size(0)

        B, N, T, _ = x.size()
        x = TensorUtils.join_dimensions(x, begin_axis=0, end_axis=2) # (B*N,T,6)
        world_from_agent = TensorUtils.repeat_by_expand_at(world_from_agent, N, dim=0)
        pos = x[..., :2]
        speed = x[..., 2]
        
        # convert x positions from agent to world frame
        if use_global_coord:
            pos = GeoUtils.transform_points_tensor(pos.to(x.device), world_from_agent)

        pos_x = pos[..., 0]
        pos_y = pos[..., 1]

        assert pos_x.shape == (B*N, T)

        if already_stopped is None:
            already_stopped = torch.zeros((B*N, T), device=x.device, dtype=torch.bool)

        robustness = self.stop_sign.get_robustness(speed, pos_x, pos_y, already_stopped, self.horizon_length, self.time_step_to_start, self.num_time_steps_to_stop, self.scale, use_until=use_until)
        loss = -torch.clip(robustness, max=0)
        # # Normalization Change: min=-1
        # loss = -torch.clip(robustness, min=-1, max=0)
        loss = loss.reshape((B, N))

        return loss

class GlobalStopSignLoss(GuidanceLoss):
    '''
    A global stop-sign example.
    Vehicles are expected to have three consecutive time steps of low speed when passing the stop sign regions.
    '''
    def __init__(self, stop_sign_pos, stop_box_dim, scale, horizon_length, time_step_to_start, num_time_steps_to_stop, action_num, low_speed_th):
        '''
        - stop_sign_pos : stop sign region center position (global frame)
        - stop_box_dim: stop box dimension (width, height)

        - scale: scale factor for smoothing out STL robust value (< 0 uses max/min, positive values closer to 0 gives more uniform gradients)

        - horizon_length: length of the horizon to use for computing STL robust value
        - time_step_to_start: time step to start using STL robust value
        - num_time_steps_to_stop: number of time steps to stop in the stop sign region

        - action_num: the number of action steps for each rollout step
        - low_speed_th: the speed threshold to use for the low speed condition
        '''
        super().__init__()

        self.stop_box_dim = torch.tensor(stop_box_dim)
        self.scale = torch.tensor(scale)

        self.horizon_length = horizon_length
        self.time_step_to_start = time_step_to_start
        self.num_time_steps_to_stop = num_time_steps_to_stop
        
        self.action_num = action_num
        self.low_speed_th = low_speed_th

        self.stop_sign_pos = torch.tensor(stop_sign_pos)

        # use world frame
        self.local_stop_sign_loss = StopSignLoss(stop_sign_pos, stop_box_dim, scale, horizon_length, time_step_to_start, num_time_steps_to_stop, action_num, low_speed_th)

        # record if vehicles have stopped in stop sign region; stl loss only applies to those which have not stopped in stog sign region.
        self.already_stopped = None

    def forward(self, x, data_batch, agt_mask=None):
        '''
        - x : the current trajectory (B, N, T, 6) where N is the number of samples and 6 is (x, y, vel, yaw, acc, yawvel)
        '''
        world_from_agent = data_batch["world_from_agent"]
        agent_hist = data_batch['agent_hist']
        if agt_mask is not None:
            x = x[agt_mask]
            world_from_agent = world_from_agent[agt_mask]
            agent_hist = agent_hist[agt_mask]
        B, N, T, _ = x.size()
        assert self.stop_box_dim.size(0) == self.stop_sign_pos.size(0) == B, f"{self.stop_box_dim.size(0)}, {self.stop_sign_pos.size(0)}, {B}"
        agent_hist = TensorUtils.repeat_by_expand_at(agent_hist, N, dim=0)


        # flag the vehicles which have just stopped in the stop sign regions
        if self.already_stopped is None:
            self.already_stopped = torch.zeros(B*N, device=x.device)

        if self.local_stop_sign_loss.stop_sign is not None:
            # determine if the vehicles were in the stop sign regions in the last time steps.
            # this should only be considered after the first rollout step.
            # x,y,vx,vy,ax,ay,sin(psi),cos(psi)
            pos_hist = agent_hist[...,-self.action_num:,:2]
            vx_hist = agent_hist[...,-self.action_num:,2]
            vy_hist = agent_hist[...,-self.action_num:,3]
            speed_hist = torch.sqrt(vx_hist**2 + vy_hist**2)
            
            # convert from agent to world frame; we check history using world frame
            if use_global_coord:
                world_from_agent_expand = TensorUtils.repeat_by_expand_at(world_from_agent, N, dim=0)
                pos_hist = GeoUtils.transform_points_tensor(pos_hist.to(x.device), world_from_agent_expand)
            else:
                raise NotImplementedError('Have not implemented local frame version yet')

            # [B*N, T]
            # print('B, N, T', B, N, T)
            # print('pos_hist[:, :, None, :].shape', pos_hist[:, :, None, :].shape)
            # print(self.local_stop_sign_loss.stop_sign.stop_box.x_min.shape)
            inclusion_mask = self.local_stop_sign_loss.stop_sign.check_inclusion(pos_hist[:, :, None, :]).reshape([B*N, self.action_num])
            # print('inclusion_mask.shape', inclusion_mask.shape)
            # [B*N,]
            outside_mask = ~inclusion_mask[:, -1]
            # [B*N, T]
            low_speed_mask = speed_hist < self.low_speed_th

            # print('speed_hist', speed_hist)
            # print('self.low_speed_th', self.low_speed_th)
            # print('inclusion_mask', inclusion_mask, )
            # print('outside_mask', outside_mask)
            # print('low_speed_mask', low_speed_mask)
            # [B*N,]
            stop_mask = torch.sum(inclusion_mask & low_speed_mask, axis=-1) >= self.num_time_steps_to_stop

            # print('stop_mask', stop_mask)

            self.already_stopped[stop_mask] = 1

            # print('1 self_already_stopped', self.already_stopped)

            # reset if outside stop sign regions
            self.already_stopped[outside_mask] = 0
            # print('2 self_already_stopped', self.already_stopped)
        

        if use_global_coord:
            self.local_stop_sign_loss.set_target(self.stop_sign_pos.to(x.device), N=N, device=x.device)
        else:
            raise NotImplementedError('Have not implemented local frame version yet')
            # # convert stop sign pos from world to agent frame
            # local_stop_sign_pos = GeoUtils.transform_points_tensor(self.stop_sign_pos[:, None].to(x.device), agent_from_world)[:, 0]
            # self.local_stop_sign_loss.set_target(local_stop_sign_pos, N=N, device=x.device)


        # only consider those vehicles which have not stopped in the stop sign regions
        # print('self.already_stopped', self.already_stopped)
        # already_stopped = self.already_stopped.reshape([B, N])
        already_stopped_expand = self.already_stopped[:, None].expand(B*N, T)
        # for debug
        # already_stopped = torch.zeros((B*N, T), device=x.device)

        loss = self.local_stop_sign_loss(x, {"world_from_agent": world_from_agent}, None, already_stopped=already_stopped_expand)
        # print('1 loss', loss)
        loss = loss * (1-self.already_stopped.reshape([B, N]))
        # print('2 loss', loss)
        return loss

class AccLimitLoss(GuidanceLoss):
    '''
    Keep accelerations below a certain limit.
    '''
    def __init__(self, acc_limit):
        '''
        - acc_limit : acceleration limit.
        '''
        super().__init__()
        self.acc_limit = acc_limit

    def forward(self, x, data_batch, agt_mask=None):
        '''
        - x : the current trajectory (B, N, T, 6) where N is the number of samples and 6 is (x, y, vel, yaw, acc, yawvel)

        - loss: (B, N)
        '''
        if agt_mask is not None:
            x = x[agt_mask]
        acc = x[..., [4]]
        acc_dev = torch.abs(acc) - self.acc_limit
        acc_loss = torch.clip(acc_dev, min=0)
        loss = torch.mean(acc_loss, dim=[-2, -1])

        return loss
        
# class AccLimitLoss(GuidanceLoss):
#     '''
#     Keep accelerations below a certain limit.
#     '''
#     def __init__(self, lon_acc_limit, lat_acc_limit):
#         '''
#         - acc_limit : acceleration limit.
#         '''
#         super().__init__()
#         self.lon_acc_limit = lon_acc_limit
#         self.lat_acc_limit = lat_acc_limit

#     def forward(self, x, data_batch, agt_mask=None):
#         '''
#         - x : the current trajectory (B, N, T, 6) where N is the number of samples and 6 is (x, y, vel, yaw, acc, yawvel)

#         - loss: (B, N)
#         '''
#         if agt_mask is not None:
#             x = x[agt_mask]
#         acc = x[..., [4]]
#         yaw = x[..., [3]]
#         # print('yaw', yaw)

#         lon_acc = torch.abs(acc * torch.cos(yaw))
#         lat_acc = torch.abs(acc * torch.sin(yaw))
#         # print('lon_acc', lon_acc)
#         # print('lat_acc', lat_acc)
#         lon_acc_dev = lon_acc - self.lon_acc_limit
#         lat_acc_dev = lat_acc - self.lat_acc_limit

#         lon_acc_loss = torch.clip(lon_acc_dev, min=0)
#         lat_acc_loss = torch.clip(lat_acc_dev, min=0)
#         # loss = torch.mean(lon_acc_loss+lat_acc_loss, dim=[-2, -1])

#         loss = torch.mean(lon_acc_loss, dim=[-2, -1])

#         return loss

class SpeedLimitLoss(GuidanceLoss):
    '''
    Keep speed below a certain limit.
    '''
    def __init__(self, speed_limit):
        '''
        - speed_limit : speed limit.
        '''
        super().__init__()
        self.speed_limit = speed_limit

    def forward(self, x, data_batch, agt_mask=None):
        '''
        - x : the current trajectory (B, N, T, 6) where N is the number of samples and 6 is (x, y, vel, yaw, acc, yawvel)

        - loss: (B, N)
        '''
        # TBD: hard-coded
        speed_scale = 5.0
        if agt_mask is not None:
            x = x[agt_mask]

        values = torch.abs(x[..., [2]]) - self.speed_limit
        loss = torch.clip(values, min=0)

        loss = torch.mean(loss, dim=[-2, -1])
        # # Normalization Change: / speed_scale, and clip to 1
        # loss = torch.mean(loss, dim=[-2, -1]) / speed_scale
        # loss = torch.clip(loss, max=1)
        return loss
    
class GPTLoss(GuidanceLoss):
    '''
    Satisfy natural language query via querying GPT.
    '''
    def __init__(self, query):
        '''
        - query : query for GPT API.
        '''
        super().__init__()
        self.query = query
        self.loss_fn = None

    def forward(self, x, data_batch, agt_mask=None):
        '''
        - x : the current trajectory (B, N, T, 6) where N is the number of samples and 6 is (x, y, vel, yaw, acc, yawvel)

        - loss: (B, N)
        '''
        from tbsim.utils.gpt_utils import query_gpt_wrapper
        from openai_key import openai_key

        context_description = ""
        if self.loss_fn is None:
            self.loss_fn = query_gpt_wrapper(openai_key, self.query, context_description)
        print(self.loss_fn)
        raise

        # self.loss_fn = KeepDistanceLoss()

        loss = self.loss_fn(x, data_batch, agt_mask)

        return loss


class LaneFollowingLoss(GuidanceLoss):
    '''
    Vehicles with indices target_inds should follow their current lanes.
    '''
    def __init__(self, target_inds=[1, 2, 3]):
        super().__init__()
        self.target_inds = target_inds

    def forward(self, x, data_batch, agt_mask=None):
        B, N, T, _ = x.shape
        if agt_mask is not None:
            x = x[agt_mask]
        # Select positions
        # (B,N,T,6) -> (B,N,T,2)
        pos_pred = x[..., :2]
        # Select yaws
        # (B,N,T,6) -> (B,N,T,1)
        yaw_pred = x[..., 3:4]

        # Get the current lane projection in agent coordinate
        # (B,N,T,2), (B,N,T,1), dict -> (B,N,T,3)
        lane_proj = get_current_lane_projection(pos_pred, yaw_pred, data_batch)

        # Compute the deviation from the lane projection
        # (B,N,T,2), (B,N,T,3) -> (B,N,T,2)
        pos_dev = pos_pred - lane_proj[..., :2]
        yaw_dev = yaw_pred - lane_proj[..., 2:3]

        # Compute the squared loss for position and yaw deviation
        # (B,N,T,2) -> (B,N,T)
        pos_loss = torch.sum(pos_dev ** 2, dim=-1)
        # (B,N,T,1) -> (B,N,T)
        yaw_loss = torch.squeeze(yaw_dev ** 2, dim=-1)

        # Combine position and yaw loss
        # (B,N,T), (B,N,T) -> (B,N,T)
        total_loss = pos_loss + yaw_loss

        # Select the loss for the target vehicles
        # (B,N,T), list -> (len(target_inds), N, T)
        target_losses = [select_agent_ind(total_loss, ind) for ind in self.target_inds]

        # Stack the losses for target vehicles
        # list -> (len(target_inds), N, T)
        target_losses = torch.stack(target_losses, dim=0)

        # Take the mean over time
        # (len(target_inds), N, T) -> (len(target_inds), N)
        target_losses = torch.mean(target_losses, dim=-1)

        # Take the mean over target vehicles
        # (len(target_inds), N) -> (N)
        loss = torch.mean(target_losses, dim=0)

        return loss

# GPT4
class KeepDistanceLoss(GuidanceLoss):
    '''
    Vehicle with index target_ind should always keep within min_distance and max_distance from vehicle with index ref_ind.
    '''
    def __init__(self, target_ind=20, ref_ind=1, min_distance=5, max_distance=15):
        super().__init__()
        self.target_ind = target_ind
        self.ref_ind = ref_ind
        self.min_distance = min_distance
        self.max_distance = max_distance

    def forward(self, x, data_batch, agt_mask=None):
        B, N, T, _ = x.shape
        if agt_mask is not None:
            x = x[agt_mask]
        # Select positions
        # (B,N,T,6) -> (B,N,T,2)
        pos_pred = x[..., :2]
        # Select yaws
        # (B,N,T,6) -> (B,N,T,1)
        yaw_pred = x[..., 3:4]
        # convert prediction from the respective agent coordinates to the world coordinate
        # (B,N,T,2), (B,N,T,1), dict -> (B,N,T,2), (B,N,T,1)
        pos_pred_world, yaw_pred_world = transform_coord_agents_to_world(pos_pred, yaw_pred, data_batch)
        # convert prediction from the world coordinate to the agent self.ref_ind coordinate
        # (B,N,T,2), (B,N,T,1), dict, int -> (B,N,T,2), (B,N,T,1)
        pos_pred_in_ref_ind, _ = transform_coord_world_to_agent_i(pos_pred_world, yaw_pred_world, data_batch, self.ref_ind)

        # select the relevant agents with index self.target_ind in the agent self.ref_ind coordinate
        # (B, N, T, 2), int -> (N, T, 2)
        pos_pred_i_in_ref_ind = select_agent_ind(pos_pred_in_ref_ind, self.target_ind)
        # select the relevant agents with index self.target_ind in the agent self.ref_ind coordinate
        # (B, N, T, 2), int -> (N, T, 2)
        pos_pred_j_in_ref_ind = select_agent_ind(pos_pred_in_ref_ind, self.ref_ind)
        
        # Compute the distance between the two vehicles
        # (N, T, 2), (N, T, 2) -> (N, T)
        distance = torch.norm(pos_pred_i_in_ref_ind - pos_pred_j_in_ref_ind, dim=-1)
        
        # Compute the deviation from the desired distance range
        # (N, T) -> (N, T)
        distance_dev_min = self.min_distance - distance
        distance_dev_max = distance - self.max_distance

        # Clip the deviations to 0 so that we only penalize the deviations outside the desired range
        # (N, T) -> (N, T)
        distance_loss_min = torch.clip(distance_dev_min, min=0)
        distance_loss_max = torch.clip(distance_dev_max, min=0)

        # Combine the losses
        # (N, T), (N, T) -> (N, T)
        distance_loss = distance_loss_min + distance_loss_max

        # Take the mean over time
        # (N, T) -> (N)
        distance_loss = distance_loss.mean(-1)

        return distance_loss

# GPT4





class KeepDistanceLoss2(GuidanceLoss):
    '''
    Vehicle with index target_ind should always keep within a distance of 10-30m from vehicle with index ref_ind.
    '''
    def __init__(self, target_ind=1, ref_ind=2, min_dist=10, max_dist=30, decay_rate=0.9):
        super().__init__()
        self.target_ind = target_ind
        self.ref_ind = ref_ind
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.decay_rate = decay_rate

    def forward(self, x, data_batch, agt_mask=None):
        B, N, T, _ = x.shape
        if agt_mask is not None:
            x = x[agt_mask]
        # Select positions
        # (B,N,T,6) -> (B,N,T,2)
        pos_pred = x[..., :2]
        # Select yaws
        # (B,N,T,6) -> (B,N,T,1)
        yaw_pred = x[..., 3:4]
        # convert prediction from the respective agent coordinates to the world coordinate
        # (B,N,T,2), (B,N,T,1) -> (B,N,T,2), (B,N,T,1)
        pos_pred_world, yaw_pred_world = transform_coord_agents_to_world(pos_pred, yaw_pred, data_batch)
        # convert prediction from the world coordinate to the agent self.ref_ind coordinate
        # (B,N,T,2), (B,N,T,1), dict, int -> (B,N,T,2), (B,N,T,1)
        pos_pred_in_ref_ind, _ = transform_coord_world_to_agent_i(pos_pred_world, yaw_pred_world, data_batch, self.ref_ind)

        # select the relevant agents with index self.target_ind in the agent self.ref_ind coordinate
        # (B, N, T, 2), int -> (N, T, 2)
        pos_pred_i_in_ref_ind = select_agent_ind(pos_pred_in_ref_ind, self.target_ind)
        # select the relevant agents with index self.ref_ind in the agent self.ref_ind coordinate
        # (B, N, T, 2), int -> (N, T, 2)
        pos_pred_j_in_ref_ind = select_agent_ind(pos_pred_in_ref_ind, self.ref_ind)

        # Compute the distance between the two agents
        # (N, T, 2), (N, T, 2) -> (N, T)
        dist = torch.norm(pos_pred_i_in_ref_ind - pos_pred_j_in_ref_ind, dim=-1)

        # Compute the deviation from the desired distance
        # (N, T) -> (N, T)
        dist_dev = torch.where(dist < self.min_dist, self.min_dist - dist, torch.where(dist > self.max_dist, dist - self.max_dist, torch.zeros_like(dist)))

        # penalize early steps more than later steps
        exp_weights = torch.tensor([self.decay_rate ** t for t in range(T)], device=x.device)
        exp_weights /= exp_weights.sum()
        dist_dev = dist_dev * exp_weights[None, :]
        # Take the mean over time
        # (N, T) -> (N)
        dist_loss = dist_dev.mean(-1)

        return dist_loss


# GPT4
class ChangeToLeftLaneLoss(GuidanceLoss):
    '''
    Vehicle with index target_ind should change to its left lane and follow it.
    '''
    def __init__(self, target_ind=6):
        super().__init__()
        self.target_ind = target_ind

    def forward(self, x, data_batch, agt_mask=None):
        B, N, T, _ = x.shape
        if agt_mask is not None:
            x = x[agt_mask]
        # Select positions
        # (B,N,T,6) -> (B,N,T,2)
        pos_pred = x[..., :2]
        # Select yaws
        # (B,N,T,6) -> (B,N,T,1)
        yaw_pred = x[..., 3:4]

        if self.global_t == 0:
            # visualize_projection = str(self.global_t)
            visualize_projection = ''
        else:
            visualize_projection = ''

        # Get the left lane projection
        # (B,N,T,2), (B,N,T,1), dict -> (B,N,T,3)
        left_lane_proj = get_left_lane_projection(pos_pred, yaw_pred, data_batch, visualize_projection=visualize_projection)

        # Select the relevant agent with index self.target_ind
        # (B,N,T,3) -> (N,T,3)
        left_lane_proj_i = select_agent_ind(left_lane_proj, self.target_ind)

        # Compute the deviation between the predicted trajectory and the left lane projection
        # (N,T,3) -> (N,T)
        pos_dev = torch.norm(left_lane_proj_i[..., :2] - pos_pred[self.target_ind], dim=-1)
        yaw_dev = torch.abs(left_lane_proj_i[..., 2] - yaw_pred[self.target_ind].squeeze(-1))

        # Combine position and yaw deviation
        # (N,T), (N,T) -> (N,T)
        total_dev = pos_dev + yaw_dev

        # Take the mean over time
        # (N,T) -> (N)
        loss = torch.mean(total_dev, dim=-1)

        return loss

# GPT4
class FrontCollisionLoss(GuidanceLoss):
    '''
    Vehicle with index target_ind should collide with the front side of vehicle with index ref_ind.
    '''
    def __init__(self, target_ind=2, ref_ind=3):
        super().__init__()
        self.target_ind = target_ind
        self.ref_ind = ref_ind

    def forward(self, x, data_batch, agt_mask=None):
        B, N, T, _ = x.shape
        if agt_mask is not None:
            x = x[agt_mask]
        # Select positions
        # (B,N,T,6) -> (B,N,T,2)
        pos_pred = x[..., :2]
        # Select yaws
        # (B,N,T,6) -> (B,N,T,1)
        yaw_pred = x[..., 3:4]
        # convert prediction from the respective agent coordinates to the world coordinate
        # (B,N,T,2), (B,N,T,1), dict -> (B,N,T,2), (B,N,T,1)
        pos_pred_world, yaw_pred_world = transform_coord_agents_to_world(pos_pred, yaw_pred, data_batch)
        # convert prediction from the world coordinate to the agent self.ref_ind coordinate
        # (B,N,T,2), (B,N,T,1), dict, int -> (B,N,T,2), (B,N,T,1)
        pos_pred_in_ref_ind, _ = transform_coord_world_to_agent_i(pos_pred_world, yaw_pred_world, data_batch, self.ref_ind)

        # select the relevant agents with index self.target_ind in the agent self.ref_ind coordinate
        # (B, N, T, 2), int -> (N, T, 2)
        pos_pred_i_in_ref_ind = select_agent_ind(pos_pred_in_ref_ind, self.target_ind)
        # select the relevant agents with index self.target_ind in the agent self.ref_ind coordinate
        # (B, N, T, 2), int -> (N, T, 2)
        pos_pred_j_in_ref_ind = select_agent_ind(pos_pred_in_ref_ind, self.ref_ind)
        
        # Compute the position deviation in x and y axis
        # (N, T, 2), (N, T, 2) -> (N, T, 2)
        pos_dev = pos_pred_j_in_ref_ind - pos_pred_i_in_ref_ind

        # We want the target vehicle to collide with the front side of the reference vehicle,
        # so we want the x-axis deviation to be close to 0 and the y-axis deviation to be positive.
        # Compute the loss for x-axis deviation
        x_dev_loss = torch.abs(pos_dev[..., 0])

        # Compute the loss for y-axis deviation
        y_dev_loss = torch.clip(-pos_dev[..., 1], min=0)

        # Combine the x and y-axis losses
        collision_loss = x_dev_loss + y_dev_loss

        # Take the mean over time
        # (N, T) -> (N)
        loss = collision_loss.mean(-1)

        return loss

# GPT4
class CollideLeftSideLoss(GuidanceLoss):
    '''
    Vehicle with index target_ind should collide the left side of vehicle with index ref_ind.
    10: 2,3
    99: 4,2
    '''
    def __init__(self, target_ind=2, ref_ind=3):
        super().__init__()
        self.target_ind = target_ind
        self.ref_ind = ref_ind

    def forward(self, x, data_batch, agt_mask=None):
        B, N, T, _ = x.shape
        if agt_mask is not None:
            x = x[agt_mask]
        # Select positions
        # (B,N,T,6) -> (B,N,T,2)
        pos_pred = x[..., :2]
        # Select yaws
        # (B,N,T,6) -> (B,N,T,1)
        yaw_pred = x[..., 3:4]
        # convert prediction from the respective agent coordinates to the world coordinate
        # (B,N,T,2), (B,N,T,1), dict -> (B,N,T,2), (B,N,T,1)
        pos_pred_world, yaw_pred_world = transform_coord_agents_to_world(pos_pred, yaw_pred, data_batch)
        # convert prediction from the world coordinate to the agent self.ref_ind coordinate
        # (B,N,T,2), (B,N,T,1), dict, int -> (B,N,T,2), (B,N,T,1)
        pos_pred_in_ref_ind, _ = transform_coord_world_to_agent_i(pos_pred_world, yaw_pred_world, data_batch, self.ref_ind)

        # select the relevant agents with index self.target_ind in the agent self.ref_ind coordinate
        # (B, N, T, 2), int -> (N, T, 2)
        pos_pred_i_in_ref_ind = select_agent_ind(pos_pred_in_ref_ind, self.target_ind)
        # select the relevant agents with index self.target_ind in the agent self.ref_ind coordinate
        # (B, N, T, 2), int -> (N, T, 2)
        pos_pred_j_in_ref_ind = select_agent_ind(pos_pred_in_ref_ind, self.ref_ind)
        
        # Compute the position deviation in both x and y axis
        # (N, T, 2), (N, T, 2) -> (N, T, 2)
        pos_dev = pos_pred_j_in_ref_ind - pos_pred_i_in_ref_ind

        # We want the target vehicle to collide with the left side of the reference vehicle,
        # so we want the x-axis deviation to be close to 0 and the y-axis deviation to be negative.
        # Compute the loss for x-axis deviation
        x_loss = torch.abs(pos_dev[..., 0])
        # x_loss = torch.clip(pos_dev[..., 0], min=0)
        # Compute the loss for y-axis deviation
        y_loss = torch.clip(pos_dev[..., 1], min=0)
        # y_loss = torch.abs(pos_dev[..., 1])
        

        # Combine the x and y losses
        # (N, T), (N, T) -> (N, T)
        pos_loss = x_loss + y_loss

        # Take the mean over time
        # (N, T) -> (N)
        pos_loss = pos_loss.mean(-1)

        return pos_loss

class FollowLaneLoss(GuidanceLoss):
    '''
    Vehicle with index target_ind should always follow its current lane.
    '''
    def __init__(self, target_ind=4, decay_rate=0.9, T=52):
        super().__init__()
        self.target_ind = target_ind
        self.decay_rate = decay_rate
        self.T = T

    def forward(self, x, data_batch, agt_mask=None):
        B, N, _, _ = x.shape
        if agt_mask is not None:
            x = x[agt_mask]
        # Select positions
        # (B,N,T,6) -> (B,N,T,2)
        pos_pred = x[..., :self.T, :2]
        # Select yaws
        # (B,N,T,6) -> (B,N,T,1)
        yaw_pred = x[..., :self.T, 3:4]

        # get the current lane of the target vehicle
        # (B,N,T,2), (B,N,T,1), dict, int -> (B,N,T,3)
        if self.global_t == 0:
            visualize_projection = str(self.global_t)
            visualize_projection = ''
        else:
            visualize_projection = ''

        lane_pred = get_current_lane_projection(pos_pred, yaw_pred, data_batch, visualize_projection=visualize_projection)

        # (B,N)
        # lane_mask = torch.max(lane_pred[..., 0], dim=-1)[0] < 8

        # (B,N,T,3) -> (B,N,T)
        pos_dev = torch.sum(torch.abs(pos_pred - lane_pred[...,:2]), dim=-1)
        # pos_dev[lane_mask] = 0.0
        # (B,N,T) -> (N,T)
        pos_dev = pos_dev[self.target_ind]
        pos_dev.unsqueeze(0).expand(B, N, self.T)
        
        # Clip the position deviation to 0 so that we only penalize the positive deviation
        # (B,N,T) -> (B,N,T)
        pos_loss = torch.clip(pos_dev, max=5)

        # penalize early steps more than later steps
        exp_weights = torch.tensor([self.decay_rate ** t for t in range(self.T)], device=x.device)
        exp_weights /= exp_weights.sum()
        pos_loss = pos_loss * exp_weights[None, :]
        # Take the mean over time
        # (B,N,T) -> (B,N)
        pos_loss = pos_loss.mean(-1)

        return pos_loss

# GPT3.5
class StayAwayLoss(GuidanceLoss):
    '''
    Vehicle with index target_ind should always stay away from vehicle with index ref_ind.
    '''
    def __init__(self, target_ind=8, ref_ind=1, min_dist=5, max_dist=15, decay_rate=0.9):
        super().__init__()
        self.target_ind = target_ind
        self.ref_ind = ref_ind
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.decay_rate = decay_rate

    def forward(self, x, data_batch, agt_mask=None):
        B, N, T, _ = x.shape
        if agt_mask is not None:
            x = x[agt_mask]
        # Select positions
        # (B,N,T,6) -> (B,N,T,2)
        pos_pred = x[..., :2]
        # Select yaws
        # (B,N,T,6) -> (B,N,T,1)
        yaw_pred = x[..., 3:4]
        # convert prediction from the respective agent coordinates to the world coordinate
        # (B,N,T,2), (B,N,T,1), dict -> (B,N,T,2), (B,N,T,1)
        pos_pred_world, yaw_pred_world = transform_coord_agents_to_world(pos_pred, yaw_pred, data_batch)
        # convert prediction from the world coordinate to the agent self.ref_ind coordinate
        # (B,N,T,2), (B,N,T,1), dict, int -> (B,N,T,2), (B,N,T,1)
        pos_pred_in_ref_ind, _ = transform_coord_world_to_agent_i(pos_pred_world, yaw_pred_world, data_batch, self.ref_ind)

        # select the relevant agents with index self.target_ind in the agent self.ref_ind coordinate
        # (B, N, T, 2), int -> (N, T, 2)
        pos_pred_i_in_ref_ind = select_agent_ind(pos_pred_in_ref_ind, self.target_ind)
        # select the relevant agents with index self.target_ind in the agent self.ref_ind coordinate
        # (B, N, T, 2), int -> (N, T, 2)
        pos_pred_j_in_ref_ind = select_agent_ind(pos_pred_in_ref_ind, self.ref_ind)
        
        # Compute the distance between the two agents
        # (N, T, 2), (N, T, 2) -> (N, T)
        dist = torch.norm(pos_pred_j_in_ref_ind - pos_pred_i_in_ref_ind, dim=-1)
        
        # Compute the deviation from the desired distance
        # (N, T) -> (N, T)
        dist_dev = dist - self.max_dist
        
        # Clip the deviation to 0 so that we only penalize the positive deviation
        # (N, T) -> (N, T)
        dist_loss = torch.clip(dist_dev, min=0)
        
        # Compute the deviation from the minimum distance
        # (N, T) -> (N, T)
        dist_dev_min = self.min_dist - dist
        
        # Clip the deviation to 0 so that we only penalize the positive deviation
        # (N, T) -> (N, T)
        dist_loss_min = torch.clip(dist_dev_min, min=0)
        
        # Combine the two losses
        # (N, T) -> (N, T)
        dist_loss = dist_loss + dist_loss_min
        
        # penalize early steps more than later steps
        exp_weights = torch.tensor([self.decay_rate ** t for t in range(T)], device=x.device)
        exp_weights /= exp_weights.sum()
        dist_loss = dist_loss * exp_weights[None, :]
        # Take the mean over time
        # (N, T) -> (N)
        dist_loss = dist_loss.mean(-1)

        return dist_loss


############## GUIDANCE utilities ########################

GUIDANCE_FUNC_MAP = {
    'target_speed' : TargetSpeedLoss,
    'agent_collision' : AgentCollisionLoss,
    'map_collision' : MapCollisionLoss,
    'target_pos_at_time' : TargetPosAtTimeLoss,
    'target_pos' : TargetPosLoss,
    'global_target_pos_at_time' : GlobalTargetPosAtTimeLoss,
    'global_target_pos' : GlobalTargetPosLoss,
    'social_group' : SocialGroupLoss,

    'stop_sign': StopSignLoss,
    'global_stop_sign': GlobalStopSignLoss,
    'acc_limit' : AccLimitLoss,
    'speed_limit' : SpeedLimitLoss,
    'gpt': GPTLoss,
    'gptcollision': CollisionLoss,
    'gptkeepdistance': KeepDistanceLoss,
    #by 
    #loss
    'collisiontypeLoss': CollisionTypeLoss,
    'collisiondiskLoss': CollisionLoss_disk,
    'ego_escape_loss': EgoEscapeLoss,
}

class DiffuserGuidance(object):
    '''
    Handles initializing guidance functions and computing gradients at test-time.
    '''
    def __init__(self, guidance_config_list, example_batch=None):
        '''
        - example_obs [optional] - if this guidance will only be used on a single batch repeatedly,
                                    i.e. the same set of scenes/agents, an example data batch can
                                    be passed in a used to init some guidance making test-time more efficient.
        '''
        self.num_scenes = len(guidance_config_list)
        assert self.num_scenes > 0, "Guidance config list must include list of guidance for each scene"
        self.guide_configs = [[]]*self.num_scenes
        for si in range(self.num_scenes):
            if len(guidance_config_list[si]) > 0:
                self.guide_configs[si] = [GuidanceConfig.from_dict(cur_cfg) for cur_cfg in guidance_config_list[si]]
                # initialize each guidance function
                for guide_cfg in self.guide_configs[si]:
                    print(guide_cfg.name)
                    guide_cfg.func = GUIDANCE_FUNC_MAP[guide_cfg.name](**guide_cfg.params)

                    if example_batch is not None:
                        guide_cfg.func.init_for_batch(example_batch)
        
    def init_for_batch(self, example_batch):
        '''
        Initializes this loss to be used repeatedly only for the given scenes/agents in the example_batch.
        e.g. this function could use the extents of agents or num agents in each scene to cache information
              that is used while evaluating the loss
        '''
        pass
    
    def update(self, **kwargs):
        for si in range(self.num_scenes):
            cur_guide = self.guide_configs[si]
            if len(cur_guide) > 0:
                for guide_cfg in cur_guide:
                    guide_cfg.func.update(**kwargs)

    def compute_guidance_loss(self, x_loss, data_batch, return_loss_tot_traj=False):
        '''
        Evaluates all guidance losses and total and individual values.
        - x_loss: (B, N, T, 6) the trajectory to use to compute losses and 6 is (x, y, vel, yaw, acc, yawvel)
        - data_batch : various tensors of size (B, ...) that may be needed for loss calculations

        - loss_tot_traj: bool, if True, returns the total loss over the each trajectory (B, N)
        '''
        bsize, num_samp, _, _ = x_loss.size()# (16,5,52,6) the prediction horizon is 52, 16? sample
        guide_losses = dict()
        loss_tot = 0.0
        _, local_scene_index = torch.unique_consecutive(data_batch['scene_split'], return_inverse=True)
        for si in range(self.num_scenes):
            cur_guide = self.guide_configs[si]
            if len(cur_guide) > 0:
                # mask out non-current current scene
                for gidx, guide_cfg in enumerate(cur_guide):
                    agt_mask = local_scene_index == si
                    if guide_cfg.agents is not None:
                        # mask out non-requested agents within the scene
                        cur_scene_inds = torch.nonzero(agt_mask, as_tuple=True)[0]
                        agt_mask_inds = cur_scene_inds[guide_cfg.agents]
                        agt_mask = torch.zeros_like(agt_mask)
                        agt_mask[agt_mask_inds] = True
                    # compute loss
                    cur_loss = guide_cfg.func(x_loss, data_batch,
                                            agt_mask=agt_mask)
                    indiv_loss = torch.ones((bsize, num_samp)).to(cur_loss.device) * np.nan # return indiv loss for whole batch, not just masked ones
                    indiv_loss[agt_mask] = cur_loss.detach().clone()
                    guide_losses[guide_cfg.name + '_scene_%03d_%02d' % (si, gidx)] = indiv_loss
                    loss_tot = loss_tot + torch.mean(cur_loss) * guide_cfg.weight
        return loss_tot, guide_losses



############## ITERATIVE PERTURBATION ########################
class PerturbationGuidance(object):
    """
    Guide trajectory to satisfy rules by directly perturbing it
    """
    def __init__(self, transform, transform_params, scale_traj=lambda x,y:x, descale_traj=lambda x,y:x) -> None:
        
        self.transform = transform
        self.transform_params = transform_params
        
        self.scale_traj = scale_traj
        self.descale_traj = descale_traj

        self.current_guidance = None

    def update(self, **kwargs):
        self.current_guidance.update(**kwargs)

    def set_guidance(self, guidance_config_list, example_batch=None):
        self.current_guidance = DiffuserGuidance(guidance_config_list, example_batch)
    
    def clear_guidance(self):
        self.current_guidance = None

    def perturb_actions_dict(self, actions_dict, data_batch, opt_params, num_samp=1):
        """Given the observation object, add Gaussian noise to positions and yaws

        Args:
            data_batch(Dict[torch.tensor]): observation dict

        Returns:
            data_batch(Dict[torch.tensor]): perturbed observation
        """
        x_initial = torch.cat((actions_dict["target_positions"], actions_dict["target_yaws"]), dim=-1)

        x_guidance, _ = self.perturb(x_initial, data_batch, opt_params, num_samp)
        # print('x_guidance.shape', x_guidance.shape)
        # x_guidance: [B*N, T, 3]
        actions_dict["target_positions"] = x_guidance[..., :2].type(torch.float32)
        actions_dict["target_yaws"] = x_guidance[..., 2:3].type(torch.float32)
        
        return actions_dict

    def perturb(self, x_initial, data_batch, opt_params, num_samp=1, decoder=None, return_grad_of=None):
        '''
        perturb the gradient and estimate the guidance loss w.r.t. the input trajectory
        Input:
            x_initial: [batch_size*num_samp, (num_agents), time_steps, feature_dim].  scaled input trajectory.
            data_batch: additional info.
            aux_info: additional info.
            opt_params: optimization parameters.
            num_samp: number of samples in x_initial.
            decoder: decode the perturbed variable to get the trajectory.
            return_grad_of: apply the gradient to which variable.
        '''
        assert self.current_guidance is not None, 'Must instantiate guidance object before calling'

        perturb_th = opt_params['perturb_th']
        
        x_guidance = x_initial
        # x_guidance may not have gradient enabled when BITS is used
        if not x_guidance.requires_grad:
            x_guidance.requires_grad_()

        if len(x_guidance.shape) == 4:
            with torch.enable_grad():
                BN, M, T, _ = x_guidance.shape
                B = int(BN // num_samp)
                x_guidance_reshaped = x_guidance.reshape(B, num_samp, M, T, -1).permute(0, 2, 1, 3, 4).reshape(B*M*num_samp, T, -1)
        else:
            x_guidance_reshaped = x_guidance

        if opt_params['optimizer'] == 'adam':
            opt = torch.optim.Adam([x_guidance], lr=opt_params['lr'])
        elif opt_params['optimizer'] == 'sgd':
            opt = torch.optim.SGD([x_guidance], lr=opt_params['lr'])
        else: 
            raise NotImplementedError('Optimizer not implemented')
        per_losses = dict()
        for _ in range(opt_params['grad_steps']):
            with torch.enable_grad():
                # for CVAE, we need to decode the latent
                if decoder is not None:
                    x_guidance_decoded = decoder(x_guidance_reshaped)
                else:
                    x_guidance_decoded = x_guidance_reshaped
                bsize = int(x_guidance_decoded.size(0) / num_samp)

                x_all = self.transform(x_guidance_decoded, data_batch, self.transform_params, bsize=bsize, num_samp=num_samp)

                x_loss = x_all.reshape((bsize, num_samp, -1, 6))
                tot_loss, per_losses = self.current_guidance.compute_guidance_loss(x_loss, data_batch)
            
            tot_loss.backward()
            opt.step()
            opt.zero_grad()
            if perturb_th is not None:
                with torch.no_grad():
                    x_delta = x_guidance - x_initial
                    x_delta_clipped = torch.clip(x_delta, -1*perturb_th, perturb_th)
                    x_guidance.data = x_initial + x_delta_clipped

        # print('x_guidance.data - x_initial', x_guidance.data - x_initial)

        return x_guidance, per_losses
    
    def perturb_video_diffusion(self, x_initial, data_batch, opt_params, num_samp=1, return_grad_of=None):
        '''
        video_diffusion only
        perturb the gradient and estimate the guidance loss w.r.t. the input trajectory
        Input:
            x_initial: [batch_size*num_samp, (num_agents), time_steps, feature_dim].  scaled input trajectory.
            data_batch: additional info.
            aux_info: additional info.
            opt_params: optimization parameters.
            num_samp: number of samples in x_initial.
            return_grad_of: apply the gradient to which variable.
        '''
        assert self.current_guidance is not None, 'Must instantiate guidance object before calling'

        perturb_th = opt_params['perturb_th']
        
        x_guidance = x_initial

        if len(x_guidance.shape) == 4:
            with torch.enable_grad():
                BN, M, T, _ = x_guidance.shape
                B = int(BN // num_samp)
                x_guidance = x_guidance.reshape(B, num_samp, M, T, -1).permute(0, 2, 1, 3, 4).reshape(B*M*num_samp, T, -1)

        per_losses = dict()
        for _ in range(opt_params['grad_steps']):
            with torch.enable_grad():
                bsize = int(x_guidance.size(0) / num_samp)

                x_all = self.transform(x_guidance, data_batch, self.transform_params, bsize=bsize, num_samp=num_samp)
                
                x_loss = x_all.reshape((bsize, num_samp, -1, 6))
                tot_loss, per_losses = self.current_guidance.compute_guidance_loss(x_loss, data_batch)
            
            tot_loss.backward()
            x_delta = opt_params['lr'] * return_grad_of.grad
            if x_initial.shape[-1] == 2:
                # only need the grad w.r.t noisy action
                x_delta = x_delta[..., [4,5]]

            x_guidance = x_initial + x_delta
            if perturb_th is not None:
                with torch.no_grad():
                    x_delta_clipped = torch.clip(x_delta, -1*perturb_th, perturb_th)
                    x_guidance.data = x_initial + x_delta_clipped

        return x_guidance, per_losses
    
    @torch.no_grad()
    def compute_guidance_loss(self, x_initial, data_batch, num_samp=1):
        '''
        -x_initial: [B*N, T, 2/3]
        '''
        assert self.current_guidance is not None, 'Must instantiate guidance object before calling'
        if len(x_initial.shape) == 4:
            BN, M, T, _ = x_initial.shape
            B = int(BN // num_samp)
            x_initial = x_initial.reshape(B, num_samp, M, T, -1).permute(0, 2, 1, 3, 4).reshape(B*M*num_samp, T, -1)
        bsize = int(x_initial.size(0) / num_samp)
        num_t = x_initial.size(1)

        x_initial_copy = x_initial.clone().detach()
        x_guidance = Variable(x_initial_copy, requires_grad=True)

        x_all = self.transform(x_guidance, data_batch, self.transform_params, bsize=bsize, num_samp=num_samp)
        
        x_loss = x_all.reshape((bsize, num_samp, num_t, 6))

        tot_loss, per_losses = self.current_guidance.compute_guidance_loss(x_loss, data_batch)

        return tot_loss, per_losses