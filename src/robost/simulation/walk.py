"""Torque-limited floating-base crawl, MIT Cheetah swing + RS02 inverse-dynamics QP.

This is NOT the complete MIT locomotion stack. Only its unmodified swing generator
is linked; support planning and MuJoCo whole-body QP are the RS02 adapter.
"""
from __future__ import annotations
import argparse
import ctypes
import json
import hashlib
import math
import os
import time
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import mujoco
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation
from robost.paths import BUILD
from robost.simulation.mpc import Terrain, build_scene, LEGS, RADIUS, ROOT


def joint_reference(pos,rot,feet,bounds):
    qdes=[]
    for i in range(4):
        hip=np.array([.26 if i<2 else -.26,-.06 if i in (0,2) else .06,0])
        p=rot.T@(feet[i]-pos)-hip
        offset=-.0890776 if i in (0,2) else .0890789
        zleg=-np.sqrt(max(.001,p[1]**2+p[2]**2-offset**2))
        roll=(np.arctan2(p[2],p[1])-np.arctan2(zleg,offset)+np.pi)%(2*np.pi)-np.pi
        length=np.clip(np.hypot(p[0],zleg),.02,.43829)
        a=np.arccos(np.clip((.22**2+length**2-.21839**2)/(2*.22*length),-1,1))
        knee=-np.arccos(np.clip((length**2-.22**2-.21839**2)/(2*.22*.21839),-1,1))
        qdes.extend([roll,a-np.arctan2(p[0],-zleg),knee])
    return np.clip(qdes,bounds[:,0]+.02,bounds[:,1]-.02)


class Swing:
    def __init__(self):
        if not (BUILD/'libcheetah_swing.so').is_file():
            from robost.cli.build_swing import build
            build()
        self.lib = ctypes.CDLL(str(BUILD / 'libcheetah_swing.so'))
        arr = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        self.lib.cheetah_swing.argtypes = [arr, arr, ctypes.c_double, ctypes.c_double,
                                         ctypes.c_double, arr]
        self.lib.cheetah_swing.restype = None

    def sample(self, start, end, phase, duration, clearance=.05):
        # Raise, translate, lower. This keeps the toe above a vertical stair face
        # before horizontal travel. Every segment is the original cubic generator.
        top = max(start[2], end[2]) + clearance
        way = [start, np.array([start[0], start[1], top]),
               np.array([end[0], end[1], top]), end]
        breaks = [0, .3, .7, 1.]
        k = min(2, int(np.searchsorted(breaks[1:], phase, side='right')))
        width = breaks[k+1] - breaks[k]
        a, b = np.array(way[k], dtype=float), np.array(way[k+1], dtype=float)
        out = np.zeros(9)
        # For monotonic vertical segments override z using a scalar cubic. The
        # upstream generator's xy curves still provide the horizontal trajectory.
        u = np.clip((phase-breaks[k])/width, 0, 1)
        self.lib.cheetah_swing(a, b, 0., float(u), duration*width, out)
        h = duration*width
        out[2] = a[2] + (b[2]-a[2])*(3*u*u-2*u*u*u)
        out[5] = (b[2]-a[2])*6*u*(1-u)/h
        out[8] = (b[2]-a[2])*(6-12*u)/(h*h)
        return out[:3], out[3:6], out[6:]


class WholeBody:
    """Full rigid-body dynamics, friction pyramids and actuator bounds in one QP.

    Variables [qdd(18), ground force(12)]. Six floating-base equations are hard
    equalities; joint rows imply torque and are bounded by physical motor limits.
    """
    def __init__(self, m, d):
        self.m, self.d = m, d
        self.ids = [m.body(l+'_foot').id for l in LEGS]
        self.base = m.body('base').id
        self.q0 = d.qpos[7:].copy()
        self.limit = np.tile([17., 17., 25.2], 4)
        self.failures = 0
        self.calls = 0
        self.last = None
        self.lastJ = None
        self.lastC = None
        self.max_residual = 0.
        self.last_qdes=d.qpos[7:].copy()
        self.feedback_clips=0
        self.mode='wbc'
        self.control_calls=0
        self.last_com_jac=None

    def control(self, pos, rot, feet, vel, acc, stance, dt):
        m, d = self.m, self.d
        self.control_calls+=1
        nv = m.nv
        J = np.zeros((12, nv)); contactJ=np.zeros_like(J); jr = np.zeros((3, nv))
        for i, body in enumerate(self.ids):
            mujoco.mj_jacBody(m, d, J[3*i:3*i+3], jr, body)
            mujoco.mj_jac(m,d,contactJ[3*i:3*i+3],jr,d.xpos[body]-np.array([0,0,RADIUS]),body)
        jointids=np.array([m.joint(f'{l}_{p}_joint').id for l in LEGS for p in ('hip','thigh','calf')])
        bounds=m.jnt_range[jointids]
        qdes=joint_reference(pos,rot,feet,bounds)
        vdes=np.clip((qdes-self.last_qdes)/dt,-6,6);self.last_qdes=qdes
        if self.mode=='static-pd':
            support=np.flatnonzero(stance)
            positions=d.xpos[self.ids][support]
            allocation=np.vstack([np.ones(len(support)),positions[:,:2].T])
            weights=np.linalg.lstsq(allocation,np.r_[1,d.subtree_com[self.base,:2]],rcond=None)[0]
            weights=np.maximum(weights,0);weights/=max(weights.sum(),1e-9)
            force=np.zeros((4,3));force[support,2]=weights*m.body_mass.sum()*9.81
            tau=d.qfrc_bias[6:]-J[:,6:].T@force.ravel()+150*(qdes-d.qpos[7:])+3*(vdes-d.qvel[6:])
            self.feedback_clips+=int(np.sum(np.abs(tau)>self.limit))
            d.ctrl[:]=np.clip(tau,-self.limit,self.limit)
            return True
        drift = np.zeros(12) if self.lastJ is None else ((J-self.lastJ)/dt)@d.qvel
        self.lastJ = J.copy()
        drift = np.clip(drift, -100, 100)
        M = np.zeros((nv,nv)); mujoco.mj_fullM(m, d, M)
        bias = d.qfrc_bias - d.qfrc_passive
        contact_drift=np.zeros(12) if self.lastC is None else ((contactJ-self.lastC)/dt)@d.qvel
        self.lastC=contactJ.copy()
        contact_drift=np.clip(contact_drift,-100,100)
        D = np.c_[M, -contactJ.T]
        tasks=[]; targets=[]
        def task(A, b, w):
            tasks.append(A*w); targets.append(np.asarray(b)*w)
        comJ=np.zeros((3,nv));mujoco.mj_jacSubtreeCom(m,d,comJ,self.base)
        com_drift=np.zeros(3) if self.last_com_jac is None else ((comJ-self.last_com_jac)/dt)@d.qvel
        self.last_com_jac=comJ.copy()
        baseA = np.zeros((6,nv+12)); baseA[:3,:nv]=comJ;baseA[3:6,3:6]=np.eye(3)
        actualR = d.xmat[self.base].reshape(3,3)
        err = Rotation.from_matrix(actualR.T@rot).as_rotvec()
        com_desired=pos+rot@np.array([-.0111,0,-.0306])
        task(baseA, np.r_[150*(com_desired-d.subtree_com[self.base])-25*(comJ@d.qvel)-com_drift,
                            100*err-20*d.qvel[3:6]], 8.)
        for i in range(4):
            sl = slice(3*i,3*i+3)
            A = np.zeros((3,nv+12)); A[:,:nv]=J[sl]
            kp,kd=(600,45) if stance[i] else (250,30)
            desired = acc[i]+kp*(feet[i]-d.xpos[self.ids[i]])+kd*(vel[i]-J[sl]@d.qvel)-drift[sl]
            task(A, desired, .3 if stance[i] else 15.)
        posture = np.zeros((12,nv+12)); posture[:,6:nv]=np.eye(12)
        task(posture, 4*(self.q0-d.qpos[7:])-2*d.qvel[6:], .15)
        T=np.vstack(tasks); b=np.concatenate(targets)
        P=T.T@T+np.diag(np.r_[np.full(nv,.01),np.full(12,.002)])
        q=-T.T@b
        constraints=[D[:6],D[6:]]
        lower=[-bias[:6],-self.limit-bias[6:]]
        upper=[-bias[:6],self.limit-bias[6:]]
        rows=np.repeat(stance,3)
        contactA=np.zeros((int(rows.sum()),nv+12)); contactA[:,:nv]=J[rows]
        contact_b=-drift[rows]-10*(J[rows]@d.qvel)
        constraints.append(contactA);lower.append(contact_b);upper.append(contact_b)
        mu=.65
        block=np.array([[1,0,-mu],[-1,0,-mu],[0,1,-mu],[0,-1,-mu],[0,0,1.]])
        C=np.zeros((20,nv+12))
        lo=[]; hi=[]
        for i in range(4):
            C[i*5:i*5+5,nv+i*3:nv+i*3+3]=block
            lo.extend([-np.inf]*4+[0]); hi.extend([0]*4+[200 if stance[i] else 0])
        constraints.append(C); lower.append(lo); upper.append(hi)
        # Braking acceleration bounds before mechanical joint limits.
        jointids=np.array([m.joint(f'{l}_{p}_joint').id for l in LEGS for p in ('hip','thigh','calf')])
        bounds=m.jnt_range[jointids]
        Q=np.zeros((12,nv+12)); Q[:,6:nv]=np.eye(12)
        constraints.append(Q)
        lower.append(np.maximum(-300,100*(bounds[:,0]+.02-d.qpos[7:])-20*d.qvel[6:]))
        upper.append(np.minimum(300,100*(bounds[:,1]-.02-d.qpos[7:])-20*d.qvel[6:]))
        solver=osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(np.triu(P)),q=q,A=sparse.csc_matrix(np.vstack(constraints)),
                     l=np.concatenate(lower),u=np.concatenate(upper),verbose=False,
                     eps_abs=1e-4,eps_rel=1e-4,max_iter=1500,polishing=False)
        if self.last is not None: solver.warm_start(x=self.last)
        r=solver.solve(raise_error=False); self.calls+=1
        if r.info.status_val not in (1,2):
            self.failures+=1
            return False
        self.last=r.x
        self.max_residual=max(self.max_residual,float(r.info.prim_res))
        tau=D[6:]@r.x+bias[6:]
        # Kinematic whole-body reference + joint feedback, as used alongside
        # inverse-dynamics feedforward in practical quadruped WBC stacks.
        kp_joint=np.repeat(np.where(stance,80.,15.),3)
        kd_joint=np.repeat(np.where(stance,2.,.8),3)
        tau+=kp_joint*(qdes-d.qpos[7:])+kd_joint*(vdes-d.qvel[6:])
        self.feedback_clips+=int(np.sum(np.abs(tau)>self.limit))
        d.ctrl[:]=np.clip(tau,-self.limit,self.limit)
        return True


class Crawl:
    def __init__(self,m,d,terrain,speed):
        self.m,self.d,self.terrain,self.speed=m,d,terrain,speed
        self.wbc=WholeBody(m,d); self.swing=Swing()
        self.feet=d.xpos[self.wbc.ids].copy()
        self.order=[2,0,3,1]
        self.index=0; self.phase='settle'; self.since=0.
        self.swing_time=1.2 if terrain.kind=='flat' else 2.4
        self.shift_time=1.0
        self.step_length=min(.15,max(.04,speed*4*(self.swing_time+self.shift_time)))
        self.swing_time=max(self.swing_time,self.step_length/max(speed,1e-5)/4-self.shift_time)
        self.body=d.qpos[:3].copy(); self.rotation=np.eye(3)
        self.active=2; self.goal=self.feet[2].copy(); self.lift=self.goal.copy()
        self.steps=0; self.events=[]; self.fault=None
        self.reverse_descent=False;self.backward=False;self.turning=False
        self.turn_count=0;self.heading=0.;self.turn_center=None

    @property
    def front(self): return [2,3] if self.backward else [0,1]

    @property
    def hind(self): return [0,1] if self.backward else [2,3]

    def body_goal(self, support):
        # Nearest point inside an inset support triangle: blindly using its
        # centroid overextends the still-grounded fourth leg on stair transitions.
        pts=self.feet[support]
        center=self.feet.mean(axis=0)
        xy=center[:2].copy(); xy[0]+=.01
        if not self.backward and self.phase in ('shift','swing') and self.goal[2]<self.lift[2]-.05:
            xy+=.85*(self.goal[:2]-self.lift[:2])
        if len(pts)==3:
            tri=pts[:,:2]; centroid=tri.mean(axis=0)
            inset=.72 if not self.backward and self.phase in ('shift','swing') and self.active>=2 and self.goal[2]<self.lift[2]-.05 else .5
            tri=centroid+inset*(tri-centroid)
            bary=np.linalg.solve(np.vstack([tri.T,np.ones(3)]),np.r_[xy,1])
            if np.min(bary)<0:
                candidates=[]
                for j in range(3):
                    a,b=tri[j],tri[(j+1)%3]; v=b-a
                    candidates.append(a+np.clip((xy-a)@v/(v@v),0,1)*v)
                xy=min(candidates,key=lambda p:np.linalg.norm(p-xy)).copy()
        # Smooth sagittal attitude following front/rear support heights.
        virtual=self.feet.copy()
        if self.phase=='swing':
            anticipate=(1.0 if self.active<2 else 0.0) if not self.backward and self.goal[2]<self.lift[2]-.05 else .6
            virtual[self.active,2]=(1-anticipate)*self.feet[self.active,2]+anticipate*self.goal[2]
        plane=np.linalg.lstsq(np.c_[virtual[:,:2],np.ones(4)],virtual[:,2],rcond=None)[0]
        slope=np.clip(plane[0],-.8,.8); side=np.clip(plane[1],-.45,.45)
        z=virtual[:,2].mean()+slope*(xy[0]-center[0])+side*(xy[1]-center[1])+.34
        along=slope*np.cos(self.heading)+side*np.sin(self.heading)
        across=-slope*np.sin(self.heading)+side*np.cos(self.heading)
        rotation=Rotation.from_euler('xyz',[np.arctan(across),-np.arctan(along),self.heading]).as_matrix()
        # Knee extension is mechanically limited to -0.87856 rad, not a straight
        # 44 cm leg. Cap body height by every grounded leg's reachable sphere.
        hips=np.array([[.26,-.06,0],[.26,.06,0],[-.26,-.06,0],[-.26,.06,0]])@rotation.T
        reach=np.hypot(.39,.089078)
        bounds=[]
        reachable_feet=self.feet.copy()
        if self.phase=='swing':
            progress=min((self.d.time-self.since)/self.swing_time,1)
            reachable_feet[self.active]=self.swing.sample(self.lift,self.goal,progress,self.swing_time)[0]
        for i in range(4):
            horizontal=np.linalg.norm(xy+hips[i,:2]-reachable_feet[i,:2])
            bounds.append(reachable_feet[i,2]+np.sqrt(max(.001,reach**2-horizontal**2))-hips[i,2])
        z=min(z,min(bounds)-.008)
        return np.r_[xy,z], rotation

    def choose_step(self):
        # Ground contact is measured, never inferred from an old commanded pose.
        self.feet=self.d.xpos[self.wbc.ids].copy()
        if (self.reverse_descent and not self.backward and not self.turning
                and self.d.qpos[0]>=self.terrain.plateau_start+.45
                and np.min(self.feet[:,0])>self.terrain.plateau_start+.04):
            self.turning=True;self.turn_center=np.array([self.terrain.plateau_start+.5,0.])
        if self.turning and self.turn_count>=48:
            self.turning=False;self.backward=True;self.heading=np.pi
            self.order=[0,2,1,3];self.index=0
        if self.turning:
            self.heading=(self.turn_count//4+1)*np.pi/12
            self.active=[2,0,3,1][self.turn_count%4];self.turn_count+=1
            self.lift=self.feet[self.active].copy()
            local=np.array([.26 if self.active<2 else -.26,-.18 if self.active in (0,2) else .18,0.])
            self.goal=Rotation.from_euler('z',self.heading).apply(local)
            self.goal[:2]+=self.turn_center;self.goal[2]=self.terrain.count*self.terrain.rise+RADIUS
            self.phase='shift';self.since=self.d.time
            return
        # Do not let front feet keep climbing while rear feet are still on a
        # lower tread. Bound the fore/hind separation before choosing a foot.
        drops=[edge for edge,top in self.terrain.transitions
               if top < self.terrain.height(edge-1e-6)-.01] if self.terrain.kind=='stairs' else []
        approaching_drop=not self.backward and any(self.feet[self.front,0].min()<edge and self.feet[self.front,0].max()>edge-.18 for edge in drops)
        for _ in range(4):
            self.active=self.order[self.index%4]
            self.index+=1
            self.lift=self.feet[self.active].copy()
            self.goal=self.lift.copy(); self.goal[0]+=self.step_length
            if self.active in self.front:
                self.goal[0]=min(self.goal[0],self.feet[self.hind,0].mean()+.55)
            else:
                compact=approaching_drop or (self.backward and self.terrain.rise>.175 and self.feet[self.front,2].mean()<self.feet[self.hind,2].mean()-.05)
                self.goal[0]=min(self.goal[0],self.feet[self.front,0].mean()-(.22 if compact else .38))
            self.goal[0]=max(self.goal[0],self.lift[0])
            if self.terrain.kind=='stairs':
                for edge,top in self.terrain.transitions:
                    if self.lift[0]<edge and self.goal[0]>edge-.055:
                        if edge in drops and self.backward:
                            if self.terrain.rise>.175 and self.active in self.front and top<self.feet[self.hind,2].max()-RADIUS-self.terrain.rise-.03:
                                # Never leave the trailing pair two risers above
                                # a leading foot. Advance the trailing feet first.
                                self.goal[0]=max(self.lift[0],edge-.06)
                            elif self.goal[0]<edge+.065:
                                self.goal[0]=edge-.06 if self.lift[0]<edge-.08 else edge+.10
                        elif edge in drops:
                            if self.lift[0]<edge-.05:
                                self.goal[0]=edge-.035
                            elif self.active<2 and self.feet[2:,0].min()<edge-.36:
                                self.goal[0]=self.lift[0]
                            elif self.active>=2 and self.feet[:2,0].min()<edge+.40:
                                # Front feet must establish a farther support
                                # before the hind leg leaves the upper edge.
                                self.goal[0]=self.lift[0]
                            else:
                                self.goal[0]=edge+(.20 if self.active<2 else .23)
                        elif self.goal[0]<edge+.065:
                            self.goal[0]=edge-.06 if self.lift[0]<edge-.08 else edge+.07
                        break
            if self.goal[0]-self.lift[0]>.012: break
        else:
            self.fault='no_feasible_footstep'
        self.goal[2]=self.terrain.height(self.goal[0])+RADIUS
        # Keep nominal lateral placement, rather than accumulating slip.
        self.goal[1]=-.18 if self.active in (0,2) else .18
        if self.backward:self.goal[1]*=-1
        self.phase='shift'; self.since=self.d.time

    def update(self,dt):
        d=self.d
        elapsed=d.time-self.since
        if self.phase=='shift' and elapsed>15:
            self.fault='support_transfer_timeout'
        stance=np.ones(4,dtype=bool)
        target=self.feet.copy(); vel=np.zeros((4,3)); acc=np.zeros((4,3))
        if self.phase=='settle' and d.time>2 and self.speed>0:
            self.feet=d.xpos[self.wbc.ids].copy()
            self.choose_step(); elapsed=0
        support=np.ones(4,dtype=bool)
        if self.phase in ('shift','swing'):
            support[self.active]=False
        pos,rot=self.body_goal(support)
        self.body+=np.clip(pos-self.body,-.10*dt,.10*dt)
        error=Rotation.from_matrix(self.rotation.T@rot).as_rotvec()
        self.rotation=self.rotation@Rotation.from_rotvec(np.clip(error,-.25*dt,.25*dt)).as_matrix()
        if self.phase=='shift' and elapsed>=self.shift_time:
            # Do not lift until the body has shifted to the support triangle.
            tri=d.xpos[self.wbc.ids][support,:2]
            bary=np.linalg.solve(np.vstack([tri.T,np.ones(3)]),np.r_[d.subtree_com[self.wbc.base,:2],1])
            if np.min(bary)>.075 and np.linalg.norm(d.qvel[:2])<.025:
                self.phase='swing'; self.since=d.time; elapsed=0
        if self.phase=='swing':
            stance[self.active]=False
            phase=min(elapsed/self.swing_time,1)
            target[self.active],vel[self.active],acc[self.active]=self.swing.sample(
                self.lift,self.goal,phase,self.swing_time)
            if elapsed>self.swing_time:
                target[self.active,2]-=min(.006,(elapsed-self.swing_time)*.01)
            if elapsed>=self.swing_time+.15:
                footid=self.wbc.ids[self.active]
                touched=False
                for contact in d.contact:
                    bodies=[self.m.geom_bodyid[contact.geom1],self.m.geom_bodyid[contact.geom2]]
                    if footid in bodies and 0 in bodies: touched=True
                close=np.linalg.norm(d.xpos[footid]-self.goal)<.035
                if touched and close:
                    self.feet[self.active]=self.goal
                    self.events.append({'t':float(d.time),'leg':LEGS[self.active],'goal':self.goal.tolist(),
                                        'actual':d.xpos[footid].tolist()})
                    self.steps+=1
                    self.choose_step()
                elif elapsed>self.swing_time+2:
                    self.fault='touchdown_timeout'
        return self.wbc.control(self.body,self.rotation,target,vel,acc,stance,dt)


class RaibertTrot:
    """RS02 package sim_gaits.py's Raibert placement / IK / joint PD baseline.

    The original RS02 gait is retained as a flat-ground comparison; it does not
    claim stair perception or stair clearance. All outputs are bounded torques.
    """
    def __init__(self,m,d,terrain,speed):
        self.m,self.d,self.terrain,self.speed=m,d,terrain,speed
        self.wbc=WholeBody(m,d)
        self.phase='trot'; self.active=0; self.steps=0; self.events=[]
        self.timing=0.

    def update(self,dt):
        d=self.d
        rpy=Rotation.from_matrix(d.xmat[self.wbc.base].reshape(3,3)).as_euler('xyz')
        roll,pitch,yaw=rpy
        ramp=np.clip((d.time-1)/1.5,0,1)
        period=.45; duty=.5
        if d.time>=1: self.timing=(self.timing+dt/period)%1
        speed=self.speed*ramp
        vfwd=d.qvel[0]*np.cos(yaw)+d.qvel[1]*np.sin(yaw)
        desired=np.zeros((4,3))
        for i in range(4):
            x=0.; z=-.34
            if d.time>=1:
                phase=(self.timing+[0,.5,.5,0][i])%1
                length=speed*period*duty
                td=length/2+.04*(vfwd-speed)
                if phase<duty:
                    x=td-length*phase/duty
                else:
                    u=(phase-duty)/(1-duty)
                    x=td-length+length*(u-np.sin(2*np.pi*u)/(2*np.pi))
                    z+=.06*ramp*np.sin(np.pi*u)
                hx=.26 if i<2 else -.26; hy=-.149078 if i in (0,2) else .149079
                z-=.6*(hx*np.sin(pitch)-hy*np.sin(roll))
            length=np.clip(np.hypot(x,z),.01,.43829)
            a=np.arccos(np.clip((.22**2+length**2-.21839**2)/(2*.22*length),-1,1))
            g=np.arccos(np.clip((.22**2+.21839**2-length**2)/(2*.22*.21839),-1,1))
            desired[i]=[0,a-np.arctan2(x,-z),-(np.pi-g)]
        tau=150*(desired.ravel()-d.qpos[7:])-3*d.qvel[6:]
        self.wbc.control_calls+=1;self.wbc.feedback_clips+=int(np.sum(np.abs(tau)>self.wbc.limit))
        d.ctrl[:]=np.clip(tau,-self.wbc.limit,self.wbc.limit)
        return True


def main():
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--terrain',choices=['flat','stairs'],default='flat')
    ap.add_argument('--step-height-cm',type=int,choices=[15,20],default=15)
    ap.add_argument('--speed',type=float,default=None,help='m/s command; default flat 0.25 / stairs 0.04. Crawl safety limits may slow it further')
    ap.add_argument('--gait',choices=['auto','crawl','trot'],default='auto',help='auto: flat trot, stairs crawl')
    ap.add_argument('--duration',type=float,default=None,help='maximum simulated seconds, flat 30 / stairs 600')
    ap.add_argument('--headless',action='store_true')
    ap.add_argument('--video',action='store_true')
    ap.add_argument('--output',type=Path,default=ROOT/'runs'/'walk')
    ap.add_argument('--resume-report',type=Path,help='diagnostic segment only: initialize from a saved shift-phase pose')
    ap.add_argument('--resume-time',type=float,default=0,help='first saved shift-phase pose at/after this time')
    ap.add_argument('--swing-time',type=float,default=2.4,help='minimum stair swing time in seconds')
    ap.add_argument('--crawl-control',choices=['wbc','static-pd'],default='wbc')
    ap.add_argument('--reverse-descent',action='store_true',help='turn 180 degrees on the plateau and descend backwards (experimental)')
    args=ap.parse_args()
    if args.speed is None: args.speed=.25 if args.terrain=='flat' else .04
    if args.duration is None: args.duration=30 if args.terrain=='flat' else 600
    if not math.isfinite(args.speed) or args.speed<0 or not math.isfinite(args.duration) or args.duration<=0:
        ap.error('speed must be finite and nonnegative; duration must be finite and positive')
    if not math.isfinite(args.swing_time) or args.swing_time<1.2: ap.error('swing-time must be finite and at least 1.2 seconds')
    t=Terrain(args.terrain,args.step_height_cm)
    m,d=build_scene(t,args.output)
    resumed=None
    if args.resume_report:
        saved=json.loads(args.resume_report.read_text())
        if saved['terrain']!=args.terrain or saved['step_height_cm']!=args.step_height_cm:
            ap.error('resume terrain and stair height must match')
        row=next((row for row in saved['log'] if row['t']>=args.resume_time and row['phase']=='shift' and 'qpos' in row),None)
        if row is None: ap.error('no suitable recorded shift-phase pose for resume')
        d.qpos[:]=row['qpos'];d.qvel[:]=0;mujoco.mj_forward(m,d)
        resumed={'report':str(args.resume_report),'time':row['t'],'velocity_reset':True}
    gait=('trot' if args.terrain=='flat' else 'crawl') if args.gait=='auto' else args.gait
    if args.terrain=='stairs' and gait=='trot': ap.error('trot is a flat-ground baseline only; use crawl for stairs')
    ctl=(Crawl if gait=='crawl' else RaibertTrot)(m,d,t,args.speed)
    ctl.wbc.mode=args.crawl_control
    if gait=='crawl':ctl.reverse_descent=args.reverse_descent and args.terrain=='stairs'
    if gait=='crawl' and args.terrain=='stairs':ctl.swing_time=max(ctl.swing_time,args.swing_time)
    viewer=None; renderer=None; writer=None
    opt=mujoco.MjvOption(); opt.geomgroup[:]=[1,1,1,0,0,0]
    cam=mujoco.MjvCamera(); cam.azimuth=100; cam.elevation=-18; cam.distance=2.
    if not args.headless:
        from mujoco import viewer as mjviewer
        viewer=mjviewer.launch_passive(m,d)
        viewer.opt.geomgroup[:]=opt.geomgroup
        viewer.cam.azimuth=cam.azimuth; viewer.cam.elevation=cam.elevation; viewer.cam.distance=cam.distance
    if args.video:
        import imageio.v2 as imageio
        renderer=mujoco.Renderer(m,480,800)
        writer=imageio.get_writer(str(args.output/f'{t.output_name}_walk.mp4'),fps=30)
    log=[]; stop=None; nextlog=0.; nextframe=0.; wall=time.perf_counter()
    peak=np.zeros(12); square=np.zeros(12); maxspeed=np.zeros(12); n=0; nonfoot=0
    motor_ratio=np.tile([1.,1.,1.48],4)
    over_rated=np.zeros(12); contact_pairs={}; joint_min_margin=np.full(12,np.inf)
    jointids=[m.joint(f'{l}_{p}_joint').id for l in LEGS for p in ('hip','thigh','calf')]
    jointbounds=m.jnt_range[jointids]
    control_dt=.004; nextcontrol=0.;finish_since=None
    try:
        while d.time<args.duration:
            tick=time.perf_counter()
            if d.time>=nextcontrol-1e-9:
                if not ctl.update(control_dt): stop='qp_failure'; break
                if getattr(ctl,'fault',None): stop=ctl.fault; break
                nextcontrol+=control_dt
            mujoco.mj_step(m,d)
            if not np.isfinite(d.qpos).all(): stop='nonfinite'; break
            rpy=Rotation.from_matrix(d.xmat[ctl.wbc.base].reshape(3,3)).as_euler('xyz')
            if abs(rpy[0])>.75 or abs(rpy[1])>1.0: stop='attitude'; break
            peak=np.maximum(peak,np.abs(d.actuator_force)); square+=d.actuator_force**2
            over_rated+=(np.abs(d.actuator_force)/motor_ratio>6)*m.opt.timestep
            joint_min_margin=np.minimum(joint_min_margin,np.minimum(d.qpos[7:]-jointbounds[:,0],jointbounds[:,1]-d.qpos[7:]))
            maxspeed=np.maximum(maxspeed,np.abs(d.qvel[6:])); n+=1
            for c in d.contact:
                bodies=[m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]]
                if 0 in bodies and any(b!=0 and b not in ctl.wbc.ids for b in bodies):
                    nonfoot+=1
                    pair='/'.join(m.body(b).name for b in bodies)
                    entry=contact_pairs.setdefault(pair,{'samples':0,'min_distance_m':0.})
                    entry['samples']+=1;entry['min_distance_m']=min(entry['min_distance_m'],float(c.dist))
            if d.time>=nextlog:
                log.append({'t':float(d.time),'xyz':d.qpos[:3].tolist(),'rpy':rpy.tolist(),
                            'feet':d.xpos[ctl.wbc.ids].tolist(),'qpos':d.qpos.tolist(),
                            'phase':ctl.phase,'active':LEGS[ctl.active]})
                nextlog+=.1
            if d.time>=nextframe:
                cam.lookat[:]=d.qpos[:3]
                if renderer:
                    renderer.update_scene(d,camera=cam,scene_option=opt); writer.append_data(renderer.render())
                if viewer:
                    if not viewer.is_running(): stop='viewer_closed'; break
                    viewer.cam.lookat[:]=cam.lookat; viewer.sync()
                nextframe+=1/30
            if args.terrain=='stairs':
                final_feet=d.xpos[ctl.wbc.ids]
                ready=(np.min(final_feet[:,0])>t.end+.15 and np.max(np.abs(final_feet[:,2]-RADIUS))<.01
                       and np.max(np.abs(rpy[:2]))<.2 and np.linalg.norm(d.qvel[:3])<.08)
                finish_since=(d.time if finish_since is None else finish_since) if ready else None
                if finish_since is not None and d.time-finish_since>.25:
                    stop='course_completed'; break
            if viewer: time.sleep(max(0,m.opt.timestep-(time.perf_counter()-tick)))
    finally:
        if writer: writer.close()
        if renderer: renderer.close()
        if viewer: viewer.close()
    report={'controller':('MIT swing + RS02 whole-body QP crawl' if args.crawl_control=='wbc' else 'MIT swing + RS02 static gravity/IK/PD crawl') if gait=='crawl' else 'RS02 Raibert/IK/PD trot baseline','terrain':args.terrain,
            'step_height_cm':args.step_height_cm,'command_speed_m_s':args.speed,
            'duration_s':float(d.time),'stop_reason':stop,'final_xyz':d.qpos[:3].tolist(),
            'final_qpos':d.qpos.tolist(),
            'success':bool((stop=='course_completed' if args.terrain=='stairs' else stop is None and d.qpos[0]>1 and d.qpos[2]>.25) and resumed is None),
            'segment_completed':bool(stop=='course_completed' and resumed is not None),
            'last_phase':ctl.phase,'last_active_leg':LEGS[ctl.active],
            'last_foot_goal':ctl.goal.tolist() if gait=='crawl' else None,
            'last_body_reference':ctl.body.tolist() if gait=='crawl' else None,
            'torque_peak_Nm':peak.tolist(),'torque_rms_Nm':np.sqrt(square/max(n,1)).tolist(),
            'joint_speed_peak_rad_s':maxspeed.tolist(),'nonfoot_contact_samples':nonfoot,
            'nonfoot_contact_pairs':contact_pairs,'joint_limit_min_margin_rad':joint_min_margin.tolist(),
            'motor_equivalent_peak_Nm':(peak/motor_ratio).tolist(),
            'motor_equivalent_rms_Nm':(np.sqrt(square/max(n,1))/motor_ratio).tolist(),
            'motor_above_6Nm_seconds':over_rated.tolist(),
            'hardware_pass':None,'self_collision_enabled':False,'terrain_height_is_known':True,
            'source_sha256':source_hash,
            'diagnostic_resume':resumed,
            'swing_time_s':ctl.swing_time if gait=='crawl' else None,
            'crawl_control':args.crawl_control if gait=='crawl' else None,
            'reverse_descent':args.reverse_descent,
            'feedback_torque_clipping_percent':100*ctl.wbc.feedback_clips/max(1,12*ctl.wbc.control_calls),
            'mujoco_version':mujoco.__version__,'joint_order':[f'{l}_{p}' for l in LEGS for p in ('hip','thigh','calf')],
            'qp_failures':ctl.wbc.failures,'qp_calls':ctl.wbc.calls,'qp_max_residual':ctl.wbc.max_residual,
            'warnings':[int(w.number) for w in d.warning],'steps':ctl.steps,'mass_kg':float(m.body_mass.sum()),
            'wall_seconds':time.perf_counter()-wall,'events':ctl.events,'log':log}
    (args.output/f'{t.output_name}_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('log','events')},indent=2))


if __name__=='__main__':
    main()
