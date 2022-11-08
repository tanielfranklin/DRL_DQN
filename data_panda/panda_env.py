#Adding a cuboid obstacle
from spatialgeometry import Cuboid
import math
import numpy as np
import random
import spatialmath as sm
import spatialgeometry as sg
import swift
import roboticstoolbox as rp
from roboticstoolbox.tools.urdf.urdf import Collision

class Panda_RL(object):
    
    def __init__(self,delta=0.01):
        #creating env elements
        self.panda = rp.models.Panda()
        self.obstacle = Cuboid([0.2, 0.2, 0.4], pose=sm.SE3(0.3, 0, 0.2)) 
        self.obstacle2 = Cuboid([0.2, 0.2, 0.3], pose=sm.SE3(0.3, 0, 1.2)) 
        self.obs_floor = Cuboid([2., 2., 0.01], pose=sm.SE3(0, 0, 0), color=[100,100,100,0])#"black") #color=[100,100,100,0]
        self.sig_R=0 # importance of pose
        self.sig_P=1 # importance of position
        self.step_penalty=-5
        self.collision_penalty=-10
        self.bonus_complete=100
        self.reset_j1=[-1.7,1.7]
        self.reset_j2=[-3,0.06]
        self.reset_j3=[0,3.75]
        self.render_status=False
        self.mag=10 #magnification factor
        self.renderize=False
        self.delta=delta
        self.reach_limits=lambda q,j_lim: True if (q<j_lim[0] or q>j_lim[1]) else False
        self.ceil=False #activate ceil obstacle

        # q init
        self.q_start=[0., -1.6, 0.,-2.6, 0., 0.71, 0.]
        
        #End joints positions
        j=[0.35, -0.84,  3.69] 
        self.q_goal=[0., j[0], 0.,j[1], 0., j[2], 0.]
        self.set_goal()
        self.fg=0.001 

        self.init_q=True
        
        

        #set initial position
        self.panda.q = self.panda.qz
        self.d_1=self.distance()
        self.mu_1=100
        self.sig_p=1.
        self.sig_R=1.
        
        self.f=np.sum(self.fitness())

    def reset_initial_q(self):
        self.reset_j1= [-1.6,-1.6]
        self.reset_j2= [-2.6,-2.6]
        self.reset_j3= [0.71,0.71]
        self.init_q=True
        
    def start_scene(self):
        self.scene = swift.Swift()
        self.scene.launch(realtime=True)
        self.scene.add(self.obs_floor)
        self.scene.add(self.obstacle)
        if self.ceil:
            self.scene.add(self.obstacle2)
        self.scene.add(self.panda, robot_alpha=0.6)
        self.set_end_target(self.q_goal)
        self.render_status=True
        self.renderize=True
        
    def close_scene(self):
        self.scene.close()
        self.render_status=False
        self.renderize=False
        
                
        
        
    def set_goal(self):
        self.Tg=self.panda.fkine(self.q_goal)        
        self.Rg,self.Pg=self.get_RP(self.Tg)
        
    def get_state(self):
        return np.array([self.panda.q[1],self.panda.q[3],self.panda.q[5]])
        
            
        
    def get_current_RP(self):
        T=self.panda.fkine(self.panda.q)
        R,P=self.get_RP(T)
        return R,P
            
        
    def get_RP(self,T):
        R=np.array(T)[0:3,0:3]
        P=np.array(T)[0:3,3:]    
        return R,P
    
    def fitness(self):
        R,_=self.get_current_RP()        
        fit_p=self.sig_p*self.distance()
        fit_R=self.sig_R*math.acos(((R@self.Rg.T).trace()-1)/2)
        
        return fit_p,fit_R

        
    def set_end_target(self,q):        
        #Add to workspace
        Tg = self.panda.fkine(q)
        axes=sg.Axes(length=0.1,pose=Tg)
        self.scene.add(axes)
        
        
    def reset(self):
        #j1 range -1.7 a 1.7
        #j2 range 0.0 a -3.
        #j3 range 0.0 a 3.7

        
        j1=self.reset_j1
        j2=self.reset_j2 #[0.0, -3.]
        j3=self.reset_j3 #[0.0, 3.7]
        
        collision=True

        if not self.init_q:
        
            #initial states without collisions
            while collision:
                self.panda.q[1]=round(random.uniform(j1[0],j1[1]),2)
                self.panda.q[3]=round(random.uniform(j2[0],j2[1]),2)
                self.panda.q[5]=round(random.uniform(j3[0],j3[1]),2)
                if self.renderize:
                    self.scene.step()
                collision=self.detect_collision()

        else:
            self.panda.q=self.q_start
            self.scene.step()
        
        return self.get_state()
        
    def get_position(self):
        return self.get_current_RP()[1]
    
    def get_q(self):
        #get free active joints
        q=[1,3,5]
        return [self.panda.q[i] for i in q]
    def apply_action(self,a):
        self.panda.q[1]+=a[0]*self.delta
        self.panda.q[3]+=a[1]*self.delta
        self.panda.q[5]+=a[2]*self.delta
        
        
        
    def step(self,a):
        #change joint angles by delta, do nothing or -delta
        #print(a)
        info=""
        d=self.distance()
        s=self.panda.q
        self.apply_action(a)        
        info=["Running",""]
        done =False
        f_now=sum(self.fitness())
        
        if self.detect_collision():
            # next_state=s
            # next_state=np.array([self.panda.q[1],self.panda.q[3],self.panda.q[5]])
            r=self.collision_penalty
            self.apply_action(-np.array(a))

            info=["Termination","Collided"]    
        elif f_now<self.fg:
            done=True
            info=["Done","Completed"]
            r=self.bonus_complete
        else:
            r=self.reward2(f_now)
        self.f=f_now
        next_state=np.array([self.panda.q[1],self.panda.q[3],self.panda.q[5]])
        if self.renderize:
            self.scene.step()  
        return next_state,r , done,info
    
    
    
    def reach_joint_limit(self):
        #j3 range -0.08 a 3.75  #j2 range -0.07 a -3. #j1 range -1.8 a 1.76
        j_lim=[(-1.74,1.76),(-3,-0.06),(0,3.75)]
        qi=[1,3,5]
        joint=[]       
        for q,j in zip(self.get_state(),j_lim):       
            joint.append(self.reach_limits(q,j))
        return joint
             
            
    def detect_collision(self):
        obs_floor = [self.panda.links[i].collided(self.obstacle) for i in range(9)]
        # Discarding collisions among first and second links with the floor
        floor = [self.panda.links[i].collided(self.obs_floor) for i in np.arange(2,9)]
        joint_limit=self.reach_joint_limit()
        if self.ceil:
            obs_ceil = [self.panda.links[i].collided(self.obstacle2) for i in range(9)]
        else:
            obs_ceil = []
        

        detection=sum(obs_ceil+obs_floor+joint_limit+floor)
        
        if detection!=0:
            return True#, [collision, collision_floor]
        else:
            return False#, [collision, collision_floor]         
        
        
    
    def reward(self,f_now):
        r=math.atan((self.f-f_now)*math.pi/2*1/self.fg)*self.mag
        
        return r
    
    def reward2(self,f_now):
        # -5 reward for each additional step     
        r=math.atan((self.f-f_now)*math.pi/2*1/self.fg)*self.mag + self.step_penalty
        
        return r       
        


    def distance(self):   
        return np.sum(np.array(self.get_position()-self.Pg)**2)
        
    # def render(self, mode='human', close=False):
    # # Render the environment to the screen