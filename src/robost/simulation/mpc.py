"""Floating-base RS02: centroidal force MPC + diagonal trot + Cartesian swing control.

No base teleporting, external stabilizing forces, or mocap constraints. All motion
comes from twelve torque-limited joint motors and terrain contacts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

# These matrices are small: a single BLAS worker avoids thread startup overhead.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import mujoco
import numpy as np
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation

from robost.paths import ROOT, PACKAGE

LEGS = ('FR', 'FL', 'RR', 'RL')
HIP = np.array([[.26, -.149078, 0], [.26, .149079, 0],
                [-.26, -.149078, 0], [-.26, .149079, 0]])
OFFSETS = np.array([0, .5, .5, 0])
RADIUS = .02661
OUT = ROOT / 'runs' / 'mpc'


class Terrain:
    def __init__(self, kind, step_height_cm=15):
        self.kind = kind
        if step_height_cm not in (15, 20):
            raise ValueError('step_height_cm must be 15 or 20')
        self.start, self.tread, self.rise, self.count = .75, .30, step_height_cm / 100, 5
        self.output_name = f'stairs_{step_height_cm}cm_course' if kind == 'stairs' else kind
        self.landing_length = 1.0
        # Five upward risers reach the plateau; five downward risers reach ground.
        self.plateau_start = self.start + (self.count - 1) * self.tread
        self.descent_start = self.plateau_start + self.landing_length
        self.end = self.descent_start + (self.count - 1) * self.tread
        self.transitions = (
            [(self.start + i * self.tread, (i + 1) * self.rise) for i in range(self.count)]
            + [(self.descent_start + i * self.tread, (self.count - i - 1) * self.rise)
               for i in range(self.count)])
        self.sections = [(f'up_{i+1}', self.start+i*self.tread, self.tread, (i+1)*self.rise)
                         for i in range(self.count-1)]
        self.sections += [('plateau', self.plateau_start, self.landing_length, self.count*self.rise)]
        self.sections += [(f'down_{i+1}', self.descent_start+i*self.tread, self.tread,
                           (self.count-i-1)*self.rise) for i in range(self.count-1)]

    def height(self, x):
        if self.kind == 'flat':
            return 0.
        values = np.zeros_like(np.asarray(x), dtype=float)
        for edge, top in self.transitions:
            values = np.where(np.asarray(x) >= edge, top, values)
        return float(values) if values.ndim == 0 else values

    def landing(self, x):
        if self.kind == 'stairs':
            for edge, _ in self.transitions:
                if edge - .06 < x < edge + .07:
                    x = edge + .07
        return float(x)


def build_scene(terrain, output_dir=None):
    tree = ET.parse(PACKAGE / 'rs02.xml')
    root = tree.getroot()
    root.find('compiler').set('meshdir', str(PACKAGE / 'meshes'))
    root.find('option').set('timestep', '0.002')
    wb = root.find('worldbody')
    floor = wb.find("geom[@name='floor']")
    floor.set('group', '0')
    floor.set('size', '8 3 .1')
    floor.set('contype', '1'); floor.set('conaffinity', '2')
    if terrain.kind == 'stairs':
        for name, left, length, top in terrain.sections:
            shade = top / (terrain.count * terrain.rise)
            ET.SubElement(wb, 'geom', name=name, type='box',
                pos=f'{left+length/2} 0 {top/2}', size=f'{length/2} .8 {top/2}',
                rgba=f'{.38+shade*.18} {.46+shade*.18} {.52+shade*.18} 1',
                friction='1 .005 .0001', contype='1', conaffinity='2', group='0')
    for body in root.iter('body'):
        for geom in body.findall('geom'):
            if geom.get('type') == 'mesh':
                geom.set('group', '1')
            else:
                geom.set('group', '3')
                # Robot/terrain collisions on; self collision off, as package recommends.
                geom.set('contype', '2'); geom.set('conaffinity', '1')
                if body.get('name', '').endswith('_foot'):
                    geom.set('name', body.get('name') + '_collision')
                    geom.set('friction', '1 .005 .0001')
                    # Separate display sphere: foot collision stays in hidden group 3.
                    ET.SubElement(body, 'geom', type='sphere', size=str(RADIUS),
                        rgba='.08 .08 .08 1', group='1', contype='0', conaffinity='0', mass='0')
    root.remove(root.find('actuator'))
    act = ET.SubElement(root, 'actuator')
    for leg in LEGS:
        for part, lim in [('hip',17), ('thigh',17), ('calf',25.2)]:
            ET.SubElement(act, 'motor', name=f'{leg}_{part}_motor',
                joint=f'{leg}_{part}_joint', ctrlrange=f'{-lim} {lim}',
                ctrllimited='true', forcerange=f'{-lim} {lim}', forcelimited='true')
    output_dir = OUT if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f'{terrain.output_name}_scene.xml'
    tree.write(path, encoding='utf-8', xml_declaration=True)
    m = mujoco.MjModel.from_xml_path(str(path)); d = mujoco.MjData(m)
    d.qpos[2] = .36661
    for leg in LEGS:
        for part, q in [('hip',0),('thigh',.68022),('calf',-1.3664)]:
            d.qpos[m.jnt_qposadr[m.joint(f'{leg}_{part}_joint').id]] = q
    mujoco.mj_forward(m,d)
    return m,d


def skew(v):
    x,y,z=v
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])


class ForceMPC:
    """12-state linear centroidal MPC, 12 foot-force controls, friction pyramids.

    States: world position, Euler angles, world linear and angular velocity.
    Predicts N steps; applies only the first force vector, then replans.
    """
    def __init__(self, mass, inertia, dt=.04, horizon=10):
        self.mass, self.inertia, self.dt, self.N = mass, inertia, dt, horizon
        self.Q = np.diag([20,100,900, 600,600,300, 100,100,100, 20,20,20])
        self.failures=0; self.calls=0; self.solve_ms=[]; self.last=None

    def solve(self, state, feet, refs, contacts, rotation):
        tick=time.perf_counter(); n=self.N; dt=self.dt
        A=np.eye(12); A[:6,6:]=np.eye(6)*dt
        B=np.zeros((12,12))
        invI=np.linalg.inv(rotation@self.inertia@rotation.T)
        for i in range(4):
            B[6:9,3*i:3*i+3]=np.eye(3)*dt/self.mass
            B[9:12,3*i:3*i+3]=invI@skew(feet[i]-state[:3])*dt
        B[:6]=.5*dt*B[6:]
        g=np.zeros(12); g[2]=-.5*9.81*dt*dt; g[8]=-9.81*dt
        # Condensed prediction x = free + S u.
        S=np.zeros((12*n,12*n)); free=[]; x=state.copy()
        row=np.zeros((12,12*n))
        for k in range(n):
            x=A@x+g; free.extend(x)
            row=A@row; row[:,12*k:12*k+12]=B
            S[12*k:12*k+12]=row
        Q=sparse.kron(sparse.eye(n),self.Q,format='csc')
        P=S.T@(Q@S)+np.eye(12*n)*.0003
        q=S.T@(Q@(np.array(free)-refs.reshape(-1)))
        # |fx|,|fy| <= mu*fz, 0<=fz<=160; swing force = 0.
        mu=.65
        block=np.array([[1,0,-mu],[-1,0,-mu],[0,1,-mu],[0,-1,-mu],[0,0,1]])
        C=sparse.kron(sparse.eye(4*n),block,format='csc')
        lower=np.tile([-np.inf]*4+[0],4*n)
        upper=np.zeros((n,4,5)); upper[:,:,4]=contacts*160
        solver=osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(np.triu(P)),q=q,A=C,l=lower,u=upper.reshape(-1),
                     verbose=False,eps_abs=.002,eps_rel=.002,max_iter=2000,polishing=False)
        if self.last is not None: solver.warm_start(x=self.last)
        result=solver.solve(raise_error=False); self.calls+=1
        self.solve_ms.append((time.perf_counter()-tick)*1000)
        if result.info.status_val not in (1,2):
            self.failures+=1
            forces=np.zeros((4,3)); mask=contacts[0]>0
            forces[mask,2]=self.mass*9.81/max(1,mask.sum())
            return forces
        self.last=result.x
        return result.x[:12].reshape(4,3)


class Trot:
    def __init__(self,m,d,terrain,speed):
        self.m,self.d,self.terrain,self.speed=m,d,terrain,speed
        self.period=.65 if terrain.kind=='flat' else .9
        self.duty=.65 if terrain.kind=='flat' else .72
        self.ids=np.array([[m.joint(f'{l}_{p}_joint').id for p in ('hip','thigh','calf')] for l in LEGS])
        self.qadr=m.jnt_qposadr[self.ids]; self.vadr=m.jnt_dofadr[self.ids]
        self.footids=[m.body(l+'_foot').id for l in LEGS]
        self.base=m.body('base').id
        self.mpc=ForceMPC(float(m.body_mass.sum()),np.diag([.23,.65,.72]))
        self.forces=np.zeros((4,3)); self.next_mpc=0
        self.anchor=d.xpos[self.footids].copy(); self.lift=self.anchor.copy(); self.touchdown=self.anchor.copy()
        self.was_stance=np.ones(4,dtype=bool)
        self.xref=0.; self.zref=.36661
        self.jac=np.zeros((3,m.nv)); self.jacr=np.zeros_like(self.jac)
        self.peak=np.zeros(12); self.saturation=0; self.samples=0

    def schedule(self,t):
        if t<1.5: return np.ones(4,dtype=bool),np.zeros(4)
        phase=((t-1.5)/self.period+OFFSETS)%1
        return phase<self.duty,phase

    def update(self,dt):
        m,d=self.m,self.d; t=d.time
        rot=d.xmat[self.base].reshape(3,3)
        angles=Rotation.from_matrix(rot).as_euler('xyz')
        vel=d.qvel[:3].copy(); omega=rot@d.qvel[3:6]
        ramp=np.clip((t-1.5)/1.5,0,1); speed=self.speed*ramp
        self.xref+=speed*dt; self.xref=np.clip(self.xref,d.qpos[0]-.08,d.qpos[0]+.10)
        stance,phase=self.schedule(t)
        feet=d.xpos[self.footids].copy()
        target=np.zeros((4,3)); targetvel=np.zeros((4,3))
        for i in range(4):
            if self.was_stance[i] and not stance[i]:
                self.lift[i]=feet[i]
                td=d.qpos[:3]+rot@HIP[i]
                td[0]+=speed*self.period*self.duty/2 + .06*(vel[0]-speed)
                td[1]+=.06*vel[1]
                td[0]=self.terrain.landing(td[0])
                td[2]=self.terrain.height(td[0])+RADIUS
                self.touchdown[i]=td
            if not self.was_stance[i] and stance[i]:
                self.anchor[i]=self.touchdown[i]
            if stance[i]:
                target[i]=self.anchor[i]
            else:
                u=(phase[i]-self.duty)/(1-self.duty)
                duration=self.period*(1-self.duty)
                blend=u*u*(3-2*u); deriv=6*u*(1-u)/duration
                delta=self.touchdown[i]-self.lift[i]
                target[i]=self.lift[i]+blend*delta
                targetvel[i]=deriv*delta
                clearance=.07 if self.terrain.kind=='flat' else .14
                target[i,2]+=clearance*np.sin(np.pi*u)
                targetvel[i,2]+=clearance*np.pi*np.cos(np.pi*u)/duration
        self.was_stance=stance.copy()
        ground=np.mean([self.terrain.height(d.qpos[0]+h[0]) for h in HIP])
        desired_z=.36661+ground
        self.zref+=np.clip(desired_z-self.zref,-.12*dt,.12*dt)
        if t>=self.next_mpc:
            state=np.r_[d.qpos[:3],angles,vel,omega]
            refs=[]; contacts=[]
            for k in range(self.mpc.N):
                future=t+(k+1)*self.mpc.dt
                refs.append(np.r_[[self.xref+speed*(k+1)*self.mpc.dt,0,self.zref],
                                  [0,0,0],[speed,0,0],[0,0,0]])
                contacts.append(self.schedule(t+k*self.mpc.dt)[0])
            self.forces=self.mpc.solve(state,feet,np.array(refs),np.array(contacts),rot)
            self.next_mpc=t+self.mpc.dt
        for i in range(4):
            mujoco.mj_jacBody(m,d,self.jac,self.jacr,self.footids[i])
            J=self.jac[:,self.vadr[i]]
            footvel=self.jac@d.qvel
            kp=120 if stance[i] else 650
            kd=8 if stance[i] else 14
            force=kp*(target[i]-feet[i])+kd*(targetvel[i]-footvel)
            if stance[i]: force-=self.forces[i]
            tau=J.T@force+d.qfrc_bias[self.vadr[i]]
            tau-=.15*d.qvel[self.vadr[i]]
            limit=np.array([17,17,25.2])
            self.saturation+=int(np.sum(np.abs(tau)>limit)); self.samples+=3
            d.ctrl[i*3:i*3+3]=np.clip(tau,-limit,limit)
        self.peak=np.maximum(self.peak,np.abs(d.ctrl))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--terrain',choices=['flat','stairs'],default='flat')
    ap.add_argument('--step-height-cm',type=int,choices=[15,20],default=15,
                    help='계단 한 칸 높이(cm), 기본 15. 평지에서는 무시')
    ap.add_argument('--duration',type=float,default=None,help='최대 실행 시간(초), 평지 12 / 계단 30')
    ap.add_argument('--speed',type=float,default=None,
                    help='전진 속도 명령(m/s), 기본 평지 0.50 / 계단 0.24; 예: --speed 0.12')
    ap.add_argument('--headless',action='store_true')
    ap.add_argument('--video',action='store_true')
    ap.add_argument('--hold',action='store_true',help='완료 후 GUI를 최종 자세로 유지')
    args=ap.parse_args()
    if args.duration is None: args.duration=12 if args.terrain=='flat' else 30
    if not math.isfinite(args.duration) or args.duration <= 0 or (args.speed is not None and
            (not math.isfinite(args.speed) or args.speed < 0)):
        ap.error('duration은 양수, speed는 0 이상이어야 합니다.')
    terrain=Terrain(args.terrain,args.step_height_cm); m,d=build_scene(terrain)
    ctl=Trot(m,d,terrain,args.speed if args.speed is not None else (.50 if args.terrain=='flat' else .24))
    option=mujoco.MjvOption(); option.geomgroup[:]=[1,1,1,0,0,0]
    cam=mujoco.MjvCamera(); cam.azimuth=135; cam.elevation=-18; cam.distance=2.2
    viewer=None; renderer=None; writer=None
    if not args.headless:
        from mujoco import viewer as mjviewer
        viewer=mjviewer.launch_passive(m,d,show_left_ui=False,show_right_ui=False)
        with viewer.lock():
            viewer.opt.geomgroup[:]=option.geomgroup
            viewer.cam.azimuth=cam.azimuth; viewer.cam.elevation=cam.elevation; viewer.cam.distance=cam.distance
    if args.video:
        import imageio.v2 as imageio
        renderer=mujoco.Renderer(m,480,800)
        writer=imageio.get_writer(str(OUT/f'{terrain.output_name}_mpc.mp4'),fps=30,codec='libx264',quality=7)
    log=[]; nextframe=0.; nextlog=0.; fallen=None; stop_reason=None; wall=time.perf_counter()
    maxspeed=np.zeros(12); maxrpy=np.zeros(3); finite=True
    try:
        while d.time<args.duration:
            start=time.perf_counter()
            ctl.update(m.opt.timestep); mujoco.mj_step(m,d)
            finite=bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
            if not finite:
                fallen=float(d.time); stop_reason='nonfinite_state'; break
            angles=Rotation.from_matrix(d.xmat[ctl.base].reshape(3,3)).as_euler('xyz')
            maxrpy=np.maximum(maxrpy,np.abs(angles))
            maxspeed=np.maximum(maxspeed,np.abs(d.qvel[ctl.vadr].reshape(-1)))
            clearance=d.qpos[2]-np.mean([terrain.height(x) for x in d.xpos[ctl.footids,0]])
            if abs(angles[0])>.85 or abs(angles[1])>.85 or clearance<.16:
                fallen=float(d.time)
                stop_reason='attitude_threshold' if max(abs(angles[0]),abs(angles[1]))>.85 else 'body_clearance_threshold'
                break
            if d.time>=nextlog:
                log.append({'t':float(d.time),'xyz':d.qpos[:3].tolist(),'rpy':angles.tolist(),
                            'feet':d.xpos[ctl.footids].tolist(),'contacts':int(d.ncon)})
                nextlog+=.05
            if d.time>=nextframe:
                cam.lookat[:]=d.qpos[:3]
                if renderer:
                    renderer.update_scene(d,camera=cam,scene_option=option)
                    frame=renderer.render(); writer.append_data(frame)
                    if nextframe==0:
                        from PIL import Image
                        Image.fromarray(frame).save(OUT/f'{terrain.output_name}_start.png')
                if viewer:
                    if not viewer.is_running(): break
                    with viewer.lock(): viewer.cam.lookat[:]=cam.lookat
                    viewer.sync()
                nextframe+=1/30
            if viewer: time.sleep(max(0,m.opt.timestep-(time.perf_counter()-start)))
        if renderer:
            from PIL import Image
            cam.lookat[:]=d.qpos[:3]
            renderer.update_scene(d,camera=cam,scene_option=option)
            Image.fromarray(renderer.render()).save(OUT/f'{terrain.output_name}_end.png')
    except KeyboardInterrupt:
        pass
    finally:
        if writer: writer.close()
        if renderer: renderer.close()
    crossed=[]
    if args.terrain=='stairs':
        for edge, top in terrain.transitions:
            crossing=next((row['t'] for row in log
                if all(p[0] > edge+.035 and p[2] > top+RADIUS-.02 for p in row['feet'])),None)
            crossed.append(crossing)
    completed=d.time>=args.duration-m.opt.timestep and fallen is None and finite
    success=completed and (d.qpos[0]>1 if args.terrain=='flat' else all(x is not None for x in crossed))
    report={'terrain':args.terrain,'controller':'centroidal force MPC + Cartesian foot control',
        'success':bool(success),'completed_duration':bool(completed),'finite_state':finite,
        'stop_reason':stop_reason,
        'duration_s':float(d.time),'command_speed_m_s':ctl.speed,'final_xyz_m':d.qpos[:3].tolist(),
        'final_feet_xyz_m':d.xpos[ctl.footids].tolist(),'all_feet_crossed_each_step_at_s':crossed,
        'course_layout':'5 up / 1 m plateau / 5 down' if args.terrain=='stairs' else 'flat',
        'step_transition_x_height_m':terrain.transitions if args.terrain=='stairs' else [],
        'course_end_x_m':terrain.end if args.terrain=='stairs' else None,
        'max_abs_roll_pitch_yaw_deg':np.degrees(maxrpy).tolist(),
        'joint_speed_peak_rad_s':maxspeed.tolist(),'mujoco_warning_counts':[int(w.number) for w in d.warning],
        'mass_kg':float(m.body_mass.sum()),'self_collision':False,
        'fell_at_s':fallen,'step_height_m':terrain.rise if args.terrain=='stairs' else 0,
        'step_depth_m':terrain.tread if args.terrain=='stairs' else 0,
        'step_width_m':1.6 if args.terrain=='stairs' else 0,
        'top_landing_length_m':terrain.landing_length if args.terrain=='stairs' else 0,
        'mpc_horizon':ctl.mpc.N,'mpc_dt_s':ctl.mpc.dt,'qp_calls':ctl.mpc.calls,
        'qp_failures':ctl.mpc.failures,'mean_mpc_ms':float(np.mean(ctl.mpc.solve_ms)),
        'torque_peak_Nm':ctl.peak.tolist(),'torque_saturation_pct':100*ctl.saturation/max(1,ctl.samples),
        'wall_seconds':time.perf_counter()-wall,'log':log}
    (OUT/f'{terrain.output_name}_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='log'},indent=2),flush=True)
    if viewer:
        try:
            if args.hold:
                print('최종 자세 유지 중. 창을 닫으면 종료합니다.',flush=True)
                while viewer.is_running(): viewer.sync(); time.sleep(1/60)
        except KeyboardInterrupt:
            pass
        finally:
            viewer.close(); time.sleep(.3)


if __name__=='__main__':
    main()
