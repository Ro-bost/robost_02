"""Offline collision/actuator screening of a saved RS02 run, not hardware certification."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np


def audit(report_path):
    report_path=Path(report_path)
    r=json.loads(report_path.read_text())
    scene=report_path.with_name(report_path.name.replace('_report.json','_scene.xml'))
    m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m)
    robot=(m.geom_bodyid!=0)&(m.geom_contype!=0)
    m.geom_conaffinity[robot]|=2
    pairs={};samples=0
    for row in r['log']:
        if 'qpos' not in row: continue
        d.qpos[:]=row['qpos'];d.qvel[:]=0;mujoco.mj_forward(m,d)
        samples+=1
        for c in d.contact:
            b1,b2=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
            if b1==0 or b2==0 or c.dist>=-.001: continue
            key='/'.join(sorted([m.body(b1).name,m.body(b2).name]))
            entry=pairs.setdefault(key,{'samples':0,'min_distance_m':0.,'first_t':row['t']})
            entry['samples']+=1;entry['min_distance_m']=min(entry['min_distance_m'],float(c.dist))
    result={'report':str(report_path.resolve()),'samples':samples,
            'method':'offline replay; original primitive collision shapes; penetration > 1 mm; not a live self-collision rerun',
            'self_collision_pairs':pairs,'hardware_certified':False,
            'motor_rms_over_6Nm':[i for i,v in enumerate(r.get('motor_equivalent_rms_Nm',[])) if v>6],
            'joint_limit_violations_over_0_01rad':[i for i,v in enumerate(r.get('joint_limit_min_margin_rad',[])) if v<-.01]}
    out=report_path.with_name(report_path.name.replace('_report.json','_audit.json'))
    out.write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('report',type=Path)
    print(json.dumps(audit(parser.parse_args().report),indent=2))
