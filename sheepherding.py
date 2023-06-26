import cv2 as cv
import numpy as np
from uuid import uuid4
from scipy.spatial.distance import cdist
from pprint import pprint
from numpy import linalg as LA
import random
#import matplotlib.pyplot as plt
import math
#np.random.seed(1)
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import torch
import gym

from typing import Optional, Union
import os
from gym.envs.registration import make, register, registry, spec
from shutil import rmtree
from collections import deque
import imageio
from copy import deepcopy
MAX_RESETS = 10000

class Dog:
    
    def __init__(self, **params):
        self.position = params["starting_position"]
        self.id = str(uuid4())
        self.delta_s = 0.1
        self.target_position = np.array([150,150])
        self.e= params["e"]
        self.behind_flock = params["ra"]*np.sqrt(params["N"]) 
        self.behind_sheep = params["ra"]*10 #???
        self.ra = params["ra"]
        self.N = params["N"]
        #self.step_strength = params["step_strength"]
        self.step_strength = 5
        
        self.possible_actions = [0, 1, 2, 3, 4,5,6,7,8]
        self.step_vectors = {
            0: [0,1],
            1: [1,0],
            2: [0,-1],
            3: [-1,0],
            4: [1,1],
            5: [1,-1],
            6: [-1,1],
            7: [-1,-1],
            8: [0,0],
        }

    def set_position(self, position):
        self.position = position

    # move the sheep
    # does not need to return a reward
    def step(self, action, L):        
        if not action in self.possible_actions:
            raise Exception("not possible dog action")

        move_vector = np.array(self.step_vectors[action])
        new_position = self.position + move_vector*self.step_strength

        if np.logical_or(new_position<2, new_position>(L-3)).any():            
            return False
        
        self.position = new_position
        return True

    def get_position(self):
        return self.position

class Sheep:
    
    

    def __init__(self, **params):
        self.id=params["_id"]
        self.position = params["starting_position"]
        self.n = params["n"]
        self.rs = params["rs"]
        self.ra = params["ra"]
        self.pa = params["pa"]
        self.c = params["c"]
        self.ps = params["ps"]
        self.h = params["h"]
        self.delta = params["delta"]
        self.p = params["p"]
        self.e = params["e"]
        self.inertia=np.array([0,0])

    def set_position(self, position):
        self.position = position
    
    def calc_LCM(self, sheep_positions, sheep_dists):        
        closest_sheep = sorted(sheep_dists, key=sheep_dists.get)[1:self.n+1]        
        closest_sheep_positions = [sheep_positions[sheep_id] for sheep_id in closest_sheep]        
        return np.mean(closest_sheep_positions, axis=0)

    def calc_repulsion_force_vector_from_too_close_neighborhood(self,sheep_positions,sheep_dists):
        too_close_sheep = list(filter(lambda sheep_id: sheep_dists[sheep_id]<self.ra and sheep_id != self.id, sheep_dists))
        if len(too_close_sheep)==0:
            return np.array([0,0])    
        
        Ra = [(self.position-sheep_positions[sheep_id])/(LA.norm(self.position-sheep_positions[sheep_id])) for sheep_id in too_close_sheep]
        
        return np.mean(Ra,axis=0)
        

    def calc_attraction_force_to_closest_n_sheep(self, sheep_positions, sheep_dists):
        LCM = self.calc_LCM(sheep_positions, sheep_dists)
        return LCM - self.position

    def calc_repulsion_force_vector_from_dog(self,dog_position, dog_dist):        
        if dog_dist > self.rs:
            return np.array([0,0])

        return self.position - dog_position

    def graze(self):        
        self.position += np.random.uniform(low=-1, high=1, size=2)*0.1


    # move the sheep
    # does not need to return a reward
    def step(self,sheep_positions, sheep_dists, dog_position, dog_dist, L):                                                        
        
        if dog_dist < self.rs:        
            C = self.calc_attraction_force_to_closest_n_sheep(sheep_positions, sheep_dists)
            C = C/LA.norm(C)

            Ra = self.calc_repulsion_force_vector_from_too_close_neighborhood(sheep_positions, sheep_dists)
            if not np.array_equal(Ra, np.array([0,0])):
                Ra = Ra/LA.norm(Ra)

            Rs = self.calc_repulsion_force_vector_from_dog(dog_position, dog_dist)        
            if not np.array_equal(Rs, np.array([0,0])):
                Rs = Rs/LA.norm(Rs)

            E = np.random.uniform(low=-1, high=1, size=2)*0.1
            
            H = self.h*self.inertia + self.c*C + self.pa*Ra + self.ps*Rs + self.e*E
            
            new_position = self.position + self.delta*H
            if np.logical_or(new_position<2, new_position>(L-3)).any():
                return 

            if np.isnan(new_position[0]) or np.isnan(new_position[1]):
                return 

            self.position = new_position
            self.inertia = H 
            
        else:
            #self.graze()
            self.inertia=np.array([0,0])

    def get_position(self):
        return self.position

    def set_position(self, position):
        self.position = position
    

class Sheepherding(gym.Env[np.ndarray, int]):
    

    def __init__(self, render_mode: Optional[str] = None):
        params = {
            "N":20,
            "L":200,
            "n":15,
            "rs":65,
            "ra":2,
            "pa":2,
            "c":1.05,
            "ps":0.5,
            "h":0.5,
            "delta":1.5,
            "p":0.05,
            "e":0.3,
            "delta_s":1.5,
            "goal":[30,30],
            "goal_radius":30,
            "max_steps_taken":300,
            "render_mode":False,
        }
        self.N = params["N"]                 # number of sheep
        self.L = params["L"]                 # size of the grid

        self.n = params["n"]                 # number of nearest neighbors to consider                         -> relevant for the sheep
        self.rs = params["rs"]               # sheperd detection distance                                      -> relevant for the sheep
        self.ra = params["ra"]               # agent to agent interaction distance                             -> relevant for the sheep
        self.pa = params["pa"]               # relative strength of repulsion from other agents                -> relevant for the sheep
        self.c = params["c"]                 # relative strength of attraction to the n nearest neighbors      -> relevant for the sheep
        self.ps = params["ps"]               # relative strength of repulsion from the sheperd                 -> relevant for the sheep
        self.h = params["h"]                 # relative strength of proceeding in the previous direction       -> relevant for the sheep        
        self.delta = params["delta"]         # agent displacement per time step                                -> relevant for the sheep
        self.p = params["p"]                 # probability of moving per time step while grazing               -> relevant for the sheep

        self.e = params["e"]                 # relative strength of angular noise                              -> relevant for the sheep and the sheperd

        self.delta_s = params["delta_s"]     # sheperd displacement per time step                              -> relevant for the sheperd
        
        self.sheep = None
        self.dog = None
        self.goal = params["goal"]
        self.goal_radius = params["goal_radius"]
        
        self.steps_taken=0
        self.max_steps_taken = params["max_steps_taken"]
        #self.action_space = [0,1,2,3]
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 400, 400), dtype=np.uint8)
        self.render_mode = params["render_mode"]
        self.store_video_file_name = params.get("store_video_file_name", "sheepherding_game.mp4")        
        self.step_strength = params.get("step_strength", 1)
        self.frames = []
        self.current_reward = 0
        self.done = False
        self.number_of_states_of_unherded_sheep = 0
        self.number_of_states_not_closer = 0

        self.eps = 0.001
        self.last_n_dists = deque(maxlen=5)
        self.num_of_resets = 0 
        self.hardening_scale = 100
        self.init_display()
        self.scale = self.calc_scale()        
        self.random_init()
        self.update_herded_sheep_val = 10
        
        
    def close(self):
        return

    def __str__(self):
        return "Sheepherding"

    def store_frames_in_gif(self,filename=None):                      
        imageio.mimsave(filename, self.frames) 

    def calc_scale(self):
        #print("calc_scale(): ")
        #np.emath.logn(base, array)
        val = 1 + (self.num_of_resets/self.hardening_scale)
        scale = np.log(val) * (self.L/10)        
        scale = max(scale, 5)
        scale = min(scale, 23)

        #scale = np.log(max(2,((int(self.num_of_resets)/self.hardening_scale))/2)) * (self.L/10)        
        return scale

    def calc_goal_radius(self):
        val = self.num_of_resets/self.hardening_scale
        goal_radius = 20/val
        goal_radius = max(goal_radius, 40)
        goal_radius = min(goal_radius, 100)
        return goal_radius

    def reset(self, **kwargs):
        
        self.num_of_resets += 1        
        #self.N = min(self.num_of_resets, 50)
        #self.N = min(self.num_of_resets, 50)
        #if self.num_of_resets % self.hardening_scale == 0:
        self.scale = self.calc_scale()
        self.goal_radius = self.calc_goal_radius()
        self.current_reward = 0
        self.number_of_states_of_unherded_sheep = 0
        self.number_of_states_not_closer = 0
        self.steps_taken=0
        self.random_init()
        self.frames = []
        GCM = self.calc_GCM()
        sheep_positions = [sheep.get_position() for sheep in self.sheep]    
        max_dist_from_GCM = self.ra*(self.N**(2/3))
        sheep_dists_from_centroid = cdist(sheep_positions, [GCM])
        deformation = sum([-1 if dist > max_dist_from_GCM else 0 for dist in sheep_dists_from_centroid]) 
        #obs = self.calc_centroid_based_observation_vector()
        self.sheep_have_been_herded = False
        
        self.render()                
        self.not_herded_sheep = self.calc_number_of_sheep_not_hered()
        self.GCM_to_goal_dist = self.calc_GCM_dist_to_goal()    
        self.num_of_sheep_close_to_goal = self.calc_number_of_sheep_near_goal()
        return np.array(self.frames[-1], dtype=np.uint8), {}
        
        #return obs

    # randomly initialize state
    def random_init(self):
        #np.random.seed(0)
        #random_x = np.random.uniform(low=0.4,high=0.8, size=self.N)*self.L
        #random_y = np.random.uniform(low=0.4,high=0.8, size=self.N)*self.L

        #scale = np.log((self.num_of_resets+1)/2) * (self.L+10)                        
                
        loc = np.random.uniform(low=0.5,high=0.8, size=1)[0]*self.L
        #loc = 100
        #print("random_init(): loc: "+ str(loc))
        #print("random_init(): self.scale: "+ str(self.scale))
        #print("random_init(): self.num_of_resets: "+ str(self.num_of_resets))
        #print("random_init(): self.goal_radius: "+ str(self.goal_radius))
        #random_x = np.random.normal(loc=loc,scale=self.scale, size=self.N)
        #random_y = np.random.normal(loc=loc,scale=self.scale, size=self.N)

        random_x = np.random.normal(loc=loc,scale=23, size=self.N)
        random_y = np.random.normal(loc=loc,scale=23, size=self.N)

        random_x = [max(5.0, x) for x in random_x]                
        random_x = [min(self.L - 5.0, x) for x in random_x]     
        
        random_y = [max(5.0, y) for y in random_y]                
        random_y = [min(self.L - 5.0, y) for y in random_y]                           

        self.sheep = [
            Sheep(
                _id=i, 
                starting_position=np.array([random_x[i], random_y[i]]), 
                ra=self.ra, 
                rs=self.rs, 
                n=self.n,
                pa=self.pa,
                c=self.c,
                ps=self.ps,
                h=self.h,
                delta=self.delta,
                p=self.p,
                e=self.e
            ) for i in range(0,self.N)]
        #dog_starting_position = [110, 110]
        dog_starting_position = [190, 190]
        #dog_starting_position = [50, 50]
        #dog_starting_position = [loc+10, loc+10]
        #random_dog_x = np.random.uniform(low=0.05,high=0.95, size=1)[0]*self.L
        #random_dog_y = np.random.uniform(low=0.05,high=0.95, size=1)[0]*self.L
        #dog_starting_position = [random_dog_x, random_dog_y]
        self.dog = Dog(starting_position=dog_starting_position, ra=self.ra, N=self.N, e=self.e,step_strength=2)
        #self.dog = Dog(starting_position=list(np.random.uniform(low=0,high=0.5, size=2)*self.L), ra=self.ra, N=self.N, e=self.e)
        #self.goal = list(np.random.uniform(low=0,high=0.5, size=2)*self.L)
        #self.displace_sheep(number_of_sheep_to_displace=min(10, np.log(self.num_of_resets)))
        self.displace_sheep(number_of_sheep_to_displace=10)
        self.sheep_have_been_herded = self.calc_number_of_sheep_not_hered() == 0

    def calc_GCM(self):        
        return np.mean([sheep.get_position() for sheep in self.sheep],axis=0)

    def distance(self, p1, p2):
        return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))

    #def get_sheep_out_of_group(self):
    #    sheep_positions = [sheep.get_position() for sheep in self.sheep]    
    #    
    #    GCM = self.calc_GCM()
    #    max_dist_from_GCM = self.ra*(self.N**(2/3))
    #    sheep_dists_from_centroid = cdist(sheep_positions, [GCM])
    #    return [idx for idx, dist in enumerate(sheep_dists_from_centroid) if dist > max_dist_from_GCM] # negative reward
    
    def calc_reward(self):

        max_dist_in_map = np.sqrt(self.L**2 + self.L**2)
        sheep_positions = [sheep.get_position() for sheep in self.sheep]    
        total_reward = 0
        GCM = self.calc_GCM()
        max_dist_from_GCM = self.ra*(self.N**(2/3))
        sheep_dists_from_centroid = cdist(sheep_positions, [GCM])
        sheep_dists_from_centroid = [dist[0] for dist in sheep_dists_from_centroid]
        total_reward += sum([-1 if dist > max_dist_from_GCM else 1 for dist in sheep_dists_from_centroid]) # negative reward

        deformation = total_reward                        
        
        if deformation == 0:    
                 
            GCM_to_goal_distance = cdist([GCM], [self.goal])[0][0]
            if GCM_to_goal_distance == 0:
                GCM_to_goal_distance = 0.1

            GCM_to_goal_distance = (GCM_to_goal_distance/max_dist_in_map)**(-1)

            dog_to_goal_distance = cdist([self.goal], [self.dog.get_position()])[0][0]
            if dog_to_goal_distance == 0:
                dog_to_goal_distance = 0.1
            dog_to_goal_distance = (dog_to_goal_distance/max_dist_in_map)**(-1)

            dog_to_GCM_distance = cdist([GCM], [self.dog.get_position()])[0][0]
            if dog_to_GCM_distance == 0:
                dog_to_GCM_distance = 0.1
            dog_to_GCM_distance = (dog_to_GCM_distance/max_dist_in_map)**(-1)

        return total_reward                                
    
    def calc_centroid_based_observation_vector(self):        
        sheep_pos = [sheep.get_position() for sheep in self.sheep]        
        sheep_pos_dict = {sheep.id: sheep.get_position() for sheep in self.sheep}
        sheep_centroid  = np.mean(sheep_pos,axis=0)        
        
        sheep_dists_from_centroid = cdist(sheep_pos, [sheep_centroid])
        sheep_dists_from_centroid_dict = {self.sheep[idxi].id: row[0] for idxi, row in enumerate(sheep_dists_from_centroid)}        
        farthest_sheep_from_centroid_id = max(sheep_dists_from_centroid_dict, key=sheep_dists_from_centroid_dict.get)                
        closest_sheep_from_centroid_id = min(sheep_dists_from_centroid_dict, key=sheep_dists_from_centroid_dict.get)                
        farthest_sheep_from_centroid_position = sheep_pos_dict[farthest_sheep_from_centroid_id]                
        closest_sheep_from_centroid_position = sheep_pos_dict[closest_sheep_from_centroid_id]                        

        sheep_dists_from_goal = cdist(sheep_pos, [self.goal])
        sheep_dists_from_goal_dict = {self.sheep[idxi].id: row[0] for idxi, row in enumerate(sheep_dists_from_goal)} 
        farthest_sheep_from_goal_id = max(sheep_dists_from_goal_dict, key=sheep_dists_from_goal_dict.get)                
        closest_sheep_from_goal_id = min(sheep_dists_from_goal_dict, key=sheep_dists_from_goal_dict.get)                
        farthest_sheep_from_goal_position = sheep_pos_dict[farthest_sheep_from_goal_id]                
        closest_sheep_from_goal_position = sheep_pos_dict[closest_sheep_from_goal_id]                               

        dog_position = self.dog.get_position()
        goal_position = self.goal
        obs_vector = np.concatenate([sheep_centroid, farthest_sheep_from_centroid_position, closest_sheep_from_centroid_position,farthest_sheep_from_goal_position, closest_sheep_from_goal_position, dog_position, goal_position],axis=0)
        return obs_vector


    def calc_number_of_sheep_not_hered(self):        
        sheep_positions = [sheep.get_position() for sheep in self.sheep]            
        GCM = self.calc_GCM()
        max_dist_from_GCM = self.ra*(self.N**(2/3))
        sheep_dists_from_GCM = cdist(sheep_positions, [GCM])
        sheep_dists_from_GCM = [dist[0] for dist in sheep_dists_from_GCM]
        return sum([1 if dist > max_dist_from_GCM else 0 for dist in sheep_dists_from_GCM]) 

    def displace_sheep(self, number_of_sheep_to_displace): 
        not_herded_sheep = self.calc_number_of_sheep_not_hered()
        sheep_positions = [sheep.get_position() for sheep in self.sheep]            
        GCM = self.calc_GCM()
        max_dist_from_GCM = self.ra*(self.N**(2/3))
        sheep_dists_from_GCM = cdist(sheep_positions, [GCM])
        sheep_dists_from_GCM = [dist[0] for dist in sheep_dists_from_GCM]
        displaced_sheep = not_herded_sheep
        
        for idx, (sheep, sheep_dist_from_gcm) in enumerate(zip(self.sheep, sheep_dists_from_GCM)):
            if sheep_dist_from_gcm >= max_dist_from_GCM:
                continue
            if displaced_sheep >= number_of_sheep_to_displace:
                break
            
            fi = np.random.uniform(low=0, high=2*np.pi, size=1)[0]
            current_position = sheep.get_position()
            current_position[0] += (sheep_dist_from_gcm + max_dist_from_GCM)*np.sin(fi)
            current_position[1] += (sheep_dist_from_gcm + max_dist_from_GCM)*np.cos(fi)
            self.sheep[idx].set_position(current_position)
            displaced_sheep += 1

    def calc_dog_dist_to_GCM(self):
        dog_position = self.dog.get_position()
        GCM = self.calc_GCM()
        GCM_to_dog_distance = cdist([GCM], [dog_position])[0][0]
        return GCM_to_dog_distance

    def calc_GCM_dist_to_goal(self):     
        #print("calc_GCM_dist_to_goal")   
        GCM = self.calc_GCM()
        GCM_to_goal_distance = cdist([GCM], [self.goal], "cityblock")[0][0]                        
        #print("calc_GCM_dist_to_goal: " + str(GCM_to_goal_distance))   
        return GCM_to_goal_distance

    def calc_number_of_sheep_near_goal(self):
        sheep_positions = [sheep.get_position() for sheep in self.sheep]    
        sheep_dists_from_goal = cdist(sheep_positions, [self.goal])
        num_of_sheep_close_to_goal = sum([1 if dist < self.goal_radius else 0 for dist in sheep_dists_from_goal])
        return num_of_sheep_close_to_goal 

    # move the dog in the wanted position
    # and change the position of the sheep accordingly
    # also return the next state, and reward for this action    
    #def do_action(self, action, collect=True):  


    def update_reward_based_on_herded_sheep(self, reward, done, not_herded_sheep_before, not_herded_sheep_after):      
        """
        if not_herded_sheep_after < self.not_herded_sheep:            
            reward += 10                                
            return reward, done

        if not_herded_sheep_after > self.not_herded_sheep:
            reward -= 20                    
            return reward, done
        
        if not_herded_sheep_after == 0:                
            reward += (100  + (self.max_steps_taken-self.steps_taken)**2)
            done = True
            return reward, done

        return reward, done
        
        """            
        if not_herded_sheep_after > 10:
            reward -= not_herded_sheep_after
        else:
            reward += (self.N - not_herded_sheep_after)
        return reward, done


    def update_reward_based_on_GCM_dist(self, reward, done, num_of_sheep_close_to_goal_after, GCM_to_goal_distance_after):      
        max_dist = np.sqrt(self.L**2 + self.L**2)
        GCM_sim = max_dist - GCM_to_goal_distance_after
        GCM_sim = GCM_sim / max_dist
        reward_from_GCM = ((GCM_sim*7)**2)                
        eps = 1e-2
        if GCM_to_goal_distance_after > 40:
            reward -= reward_from_GCM
        else:
            reward += reward_from_GCM
        return reward, done


    def update_reward_based_on_num_of_sheep_close_to_goal(self, reward, done, num_of_sheep_close_to_goal_before, num_of_sheep_close_to_goal_after):      
        if num_of_sheep_close_to_goal_after > 0:                
            reward += not_herded_sheep_after
        else:
            reward -= not_herded_sheep_after
        return reward, done


    def step(self, action):     
        
        self.steps_taken += 1

        GCM_to_dog_distance_before = self.calc_dog_dist_to_GCM()       
        not_herded_sheep_before = self.calc_number_of_sheep_not_hered()        
        GCM_to_goal_distance_before = self.calc_GCM_dist_to_goal()
        num_of_sheep_close_to_goal_before = self.calc_number_of_sheep_near_goal()

        if self.steps_taken % self.update_herded_sheep_val == 0:
            self.not_herded_sheep = not_herded_sheep_before
            self.GCM_to_goal_dist = GCM_to_goal_distance_before
            self.num_of_sheep_close_to_goal = num_of_sheep_close_to_goal_before

        reward = -1        
        valid_dog_step = self.dog.step(action,self.L)
        sheep_dists = self.calc_sheep_dists()
        dog_dists = self.calc_dog_dists()
        sheep_positions = {
            sheep.id:sheep.get_position() for sheep in self.sheep
        }        
        
        for sheep in self.sheep:            
            sheep.step(sheep_positions=sheep_positions, sheep_dists=sheep_dists[sheep.id], dog_position=self.dog.get_position(), dog_dist=dog_dists[sheep.id],L=self.L)

        GCM_to_dog_distance_after = self.calc_dog_dist_to_GCM()       
        not_herded_sheep_after = self.calc_number_of_sheep_not_hered()   
        GCM_to_goal_distance_after = self.calc_GCM_dist_to_goal()
        num_of_sheep_close_to_goal_after = self.calc_number_of_sheep_near_goal()
        
        if not_herded_sheep_after == 0:
            self.sheep_have_been_herded = True
        else: 
            self.sheep_have_been_herded = False

        done = False        
        reward, done = self.update_reward_based_on_herded_sheep(reward, done, not_herded_sheep_before, not_herded_sheep_after)        
        #reward, done = self.update_reward_based_on_GCM_dist(reward, done, GCM_to_goal_distance_before, GCM_to_goal_distance_after)        
        #reward, done = self.update_reward_based_on_num_of_sheep_close_to_goal(reward, done, num_of_sheep_close_to_goal_before, num_of_sheep_close_to_goal_after)        

        if not_herded_sheep_after == 0:
            reward += 500
            done = True

        if num_of_sheep_close_to_goal_after == self.N:
            reward += 1000
            done = True

        if self.steps_taken >= self.max_steps_taken:                        
            #if not_herded_sheep_after > 0:
            #reward -= 1000
            done = True

        if not valid_dog_step:             
            done = True
            #reward -= 1000

        
        self.current_reward += reward        
        self.render()                
        observation = self.frames[-1]        
        terminated = done
        truncated = None
        info = {}

        return observation, reward, terminated, truncated, info

    def init_display(self):
        self.scaling_factor = 10
        #atari_screen_shape = (210, 160, 3)
        self.atari_screen_shape = (self.L, self.L, 3)
        
        self.scaling_factor_h = self.atari_screen_shape[0]/self.L
        self.scaling_factor_w = self.atari_screen_shape[1]/self.L
                    
        self.start_display = np.zeros(self.atari_screen_shape, np.uint8)
        #self.start_display = self.render_circles(self.start_display)


    def render_circles(self, display):
        goal_x, goal_y = self.goal
        goal_x = int(goal_x*self.scaling_factor_h)
        goal_y = int(goal_y*self.scaling_factor_w)
        max_r = int(np.sqrt(self.L**2 + self.L**2))+1
        for r in range(1, max_r,2):
            circle_x = [int(goal_x + (r*np.sin(fi))) for fi in np.linspace(0, 2*np.pi, r*2)]
            circle_y = [int(goal_y + (r*np.cos(fi))) for fi in np.linspace(0, 2*np.pi, r*2)]
            color = (0,max(0,255-(1.25*r)),0)    

            for x,y in zip(circle_x, circle_y):
                if x < 0 or x > self.atari_screen_shape[0]:
                    continue

                if y < 0 or y > self.atari_screen_shape[1]:
                    continue

                point_min_x = max(x-self.scaling_factor, 0)
                point_max_x = min(x+self.scaling_factor, self.atari_screen_shape[0])

                point_min_y = max(y-self.scaling_factor,0)
                point_max_y = min(y+self.scaling_factor, self.atari_screen_shape[1])

                display[point_min_x:point_max_x, point_min_y:point_max_y,:] = color
        return display

    def render_sheep(self, display):
        GCM = self.calc_GCM()
        max_dist_from_GCM = self.ra*(self.N**(2/3))
        for sheep in self.sheep:
            sheep_x, sheep_y = sheep.get_position()            
            sheep_x = int(sheep_x*self.scaling_factor_h)
            sheep_y = int(sheep_y*self.scaling_factor_w)
            sheep_dist_from_centroid = cdist([sheep.get_position()], [GCM])[0][0]
            sheep_dist_from_goal = cdist([sheep.get_position()], [self.goal])[0][0]
            
            if sheep_dist_from_centroid > max_dist_from_GCM:
                color = (0,0,255)
            else:
                color = (255,255,255)

            if sheep_dist_from_goal <= self.goal_radius:
                color = (255,255,0)
                #continue
                      
            sheep_min_x = max(sheep_x-self.scaling_factor, 0)
            sheep_max_x = min(sheep_x+self.scaling_factor, self.atari_screen_shape[0])
            sheep_min_y = max(sheep_y-self.scaling_factor,0)
            sheep_max_y = min(sheep_y+self.scaling_factor, self.atari_screen_shape[1])

            display[sheep_min_x:sheep_max_x, sheep_min_y:sheep_max_y,:] = color                        
        return display

    def render_dog(self, display):
        dog_x, dog_y = self.dog.get_position()
        dog_x = int(dog_x*self.scaling_factor_h)
        dog_y = int(dog_y*self.scaling_factor_w)                        
        
        color = (255,0,0)                                                                                                    
        dog_min_x = max(dog_x-self.scaling_factor, 0)
        dog_max_x = min(dog_x+self.scaling_factor, self.atari_screen_shape[0])

        dog_min_y = max(dog_y-self.scaling_factor,0)
        dog_max_y = min(dog_y+self.scaling_factor, self.atari_screen_shape[1])

        display[dog_min_x:dog_max_x, dog_min_y:dog_max_y,:] = color
        return display

    def render_goal(self, display):
        goal_x, goal_y = self.goal
        goal_x = int(goal_x*self.scaling_factor_h)
        goal_y = int(goal_y*self.scaling_factor_w)

        color = (0,255,255)      
        
        goal_min_x = max(goal_x-self.scaling_factor-5, 0)
        goal_max_x = min(goal_x+self.scaling_factor+5, self.atari_screen_shape[0])        
        goal_min_y = max(goal_y-self.scaling_factor-5,0)
        goal_max_y = min(goal_y+self.scaling_factor+5, self.atari_screen_shape[1])
                                
        display[goal_min_x:goal_max_x, goal_min_y:goal_max_y,:] = color
        return display

    def render(self):
        display = deepcopy(self.start_display)
        #display = self.render_circles(display, scaling_factor, atari_screen_shape)
        display = self.render_sheep(display)
        display = self.render_dog(display)
        display = self.render_goal(display)

        self.frames.append(display)

    def calc_dog_dists(self):
        sheep_pos = [sheep.get_position() for sheep in self.sheep]
        dog_dists = cdist(sheep_pos, [self.dog.get_position()])
        return {self.sheep[idxi].id: row[0] for idxi, row in enumerate(dog_dists)}

    def calc_sheep_dists(self):        
        sheep_pos = [sheep.get_position() for sheep in self.sheep]        
        sheep_dists = cdist(sheep_pos, sheep_pos)
        return {self.sheep[idxi].id: {self.sheep[idxj].id:dist for idxj, dist in enumerate(row)} for idxi,row in enumerate(sheep_dists)}        

if __name__ == "__main__":
    register(
        id="Sheepherding-v0",
        entry_point="sheepherding:Sheepherding",
        max_episode_steps=300,
        reward_threshold=3000,
    )
    S = gym.make("Sheepherding-v0")
    total_reward = 0
    
    num_of_games_to_play = 100
    total_rewards = []
    
    rmtree("test_games")
    os.makedirs("test_games")
    for i in range(num_of_games_to_play):
        print("i: " + str(i))
        S.reset()
        done = False
        total_reward = 0
        while not done:
            action = random.sample([0,1,2,3,4,5,6,7,8],k=1)[0]            
            #action = 8
            _,reward,done,_, _ = S.step(action)
            #_, reward, done, _, _ = S.step(1)
            #_, reward, done, _, _ = S.step(3)
            #_,reward,done,_ = S.step(0)
            #_,reward,done,_ = S.step(1)
            total_reward += reward
        #S.store_frames_in_mp4("test_games/test.mp4")
        S.store_frames_in_gif("test_games/test_{}_{:.1f}.gif".format(i, total_reward))
    
        
    
    
        