"""Compare identical robot states on training and evaluation terrain geometry."""
import json
from dataclasses import dataclass
from pathlib import Path
import torch
import numpy as np
import rs02_rl_stairs as stairs
import rs02_rl as flat
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

@dataclass
class FixedCourse(stairs.CourseCfg):
    def function(self,difficulty,spec,rng):
        return super().function(13/18,spec,rng)

torch.set_num_threads(4)
arrays=[]; infos=[]
for generator in (False,True):
    cfg=stairs.make_cfg(1,42,True)
    cfg.events['reset_base'].params['pose_range']={}
    cmd=cfg.commands['twist'];cmd.rel_standing_envs=0.;cmd.ranges.lin_vel_x=(.25,.25)
    cmd.ranges.lin_vel_y=(0.,0.);cmd.ranges.ang_vel_z=(0.,0.)
    if generator:
        cfg.scene.terrain=TerrainEntityCfg(terrain_type='generator',terrain_generator=TerrainGeneratorCfg(
            size=(7.,3.),num_rows=1,sub_terrains={'course':FixedCourse()}))
    else:cfg.scene.spec_fn=lambda spec:flat.add_stair_course(spec,.15)
    env=flat.ManagerBasedRlEnv(cfg,device='cuda:0')
    command=env.command_manager.get_term('twist');command.vel_command_b[:]=torch.tensor([.25,0.,0.],device='cuda:0')
    obs,_=env.reset();command.vel_command_b[:]=torch.tensor([.25,0.,0.],device='cuda:0')
    obs=env.get_observations()
    arrays.append(obs['actor'].cpu().numpy().copy())
    model=env.sim.mj_model
    origins=env.scene.env_origins.cpu().numpy()
    infos.append({'generator':generator,'origins':origins.tolist(),'qpos':env.sim.data.qpos.cpu().numpy().tolist(),
        'terrain':[{ 'name':model.geom(i).name,'pos':(model.geom_pos[i]-origins[0]).tolist(),
            'size':model.geom_size[i].tolist(),'mask':[int(model.geom_contype[i]),int(model.geom_conaffinity[i])]} for i in range(model.ngeom) if model.geom_group[i]==0]})
    env.close()
diff=np.abs(arrays[0]-arrays[1]); result={'max_observation_difference':float(diff.max()),'differing_indices':np.flatnonzero(diff[0]>.001).tolist(),'fixed':arrays[0].tolist(),'generated':arrays[1].tolist(),'models':infos}
out=Path('output/rl/terrain_diagnosis.json');out.write_text(json.dumps(result,indent=2))
print('MAX DIFF',diff.max(),'indices',result['differing_indices'])
