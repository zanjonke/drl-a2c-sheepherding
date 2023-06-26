#!/usr/bin/env python3
import gym
import ptan
import numpy as np
import argparse
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
from gym.envs.registration import make, register, registry, spec

from lib import common
import os
from shutil import rmtree
from sheepherding import Sheepherding
import cv2 as cv


GAMMA = 0.99
LEARNING_RATE = 0.000001
ENTROPY_BETA = 0.03
BATCH_SIZE = 256
NUM_ENVS = 200

REWARD_STEPS = 6
CLIP_GRAD = 0.1

class ResnetBlock(nn.Module):

  def __init__(self, in_channels, out_channels, kernel_size, stride):
    super(ResnetBlock, self).__init__()
  
    self.conv1 = nn.Conv2d(
        in_channels=in_channels, 
        out_channels=out_channels, 
        kernel_size=kernel_size,
        stride=stride,
        padding=1,
    )
    self.bn1 = nn.BatchNorm2d(out_channels)    
    self.pool1 = nn.MaxPool2d(2,stride=2)

    self.conv2 = nn.Conv2d(
        in_channels=out_channels, 
        out_channels=out_channels, 
        kernel_size=kernel_size,
        stride=1,
        padding=1,
    )
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.pool2 = nn.MaxPool2d(2,stride=2)

    if stride == 1:
      #self.shortcut = nn.Sequential(nn.Identity())
      self.shortcut = nn.Sequential()
    else:
      # in Resnet layers conv3_1, conv4_1 and conv5_1 have a stride of 2
      # which reduces the h and w of the volume. to make the residual connection work 
      # in this scenario the image is scaled down to match thenvolume with the input.  
      self.shortcut = nn.Sequential(
          nn.Conv2d(
              in_channels,
              out_channels,
              kernel_size=1,
              stride=stride
          ),
          nn.BatchNorm2d(out_channels),
      )


  def forward(self, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = F.relu(out)
    #out = self.pool1(out)

    out = self.conv2(out)
    out = self.bn2(out)    

    out += self.shortcut(x)
    out = F.relu(out)
    #out = self.pool2(out)
    return out

class ResNet18(nn.Module):
    def __init__(self, stack_size):
        super(ResNet18, self).__init__()

        # Implement ResNet18.
        # Structure listed in https://arxiv.org/pdf/1512.03385.pdf - Table 1

        self.conv1 = nn.Conv2d(
          in_channels=stack_size, 
          out_channels=64, 
          kernel_size=7,
          stride=2,
          padding=4,
        )
        #self.bn1 = nn.BatchNorm2d(64)    
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.resnet_blocks = nn.Sequential(
              ResnetBlock(in_channels=64, out_channels=64, kernel_size=3,stride=1),
              ResnetBlock(in_channels=64, out_channels=64, kernel_size=3,stride=1),

              ResnetBlock(in_channels=64, out_channels=128, kernel_size=3,stride=2),
              ResnetBlock(in_channels=128, out_channels=128, kernel_size=3,stride=1),

              ResnetBlock(in_channels=128, out_channels=256, kernel_size=3,stride=2),
              ResnetBlock(in_channels=256, out_channels=256, kernel_size=3,stride=1),

              ResnetBlock(in_channels=256, out_channels=512, kernel_size=3,stride=2),
              ResnetBlock(in_channels=512, out_channels=512, kernel_size=3,stride=1),
        )
        #self.pool2 = nn.AvgPool2d(kernel_size=7,padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=3)
        #self.pool2 = nn.AvgPool2d(kernel_size=14)
        #self.fc_out = nn.Linear(256,num_of_classes)


    def forward(self, x):      
        #print("ResNet18.forward(): ")  
        #print("ResNet18.forward(): x.size(): " + str(x.size()))  
        out = self.conv1(x)        
        #out = self.bn1(out)
        out = F.relu(out)
        out = self.pool1(out)
        
        out = self.resnet_blocks(out)
        
        #out = torch.mean(out, dim=[2,3])        
        #print("ResNet18.forward(): out.size(): " + str(out.size()))
        out = self.pool2(out)
        out = out.view(out.size(0), -1)                
        #print("ResNet18.forward(): out.size(): " + str(out.size()))  
        #exit()
        #out = F.softmax(out)            
        return out

class SheepherdingA2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(SheepherdingA2C, self).__init__()
        self.conv = ResNet18(stack_size=input_shape[0])
        conv_out_size = self._get_conv_out(input_shape)
        #print("conv_out_size: " + str(conv_out_size))
        #exit()
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, n_actions)         
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)         
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):       
        #print("SheepherdingA2C.forward(): ")

        #sample = x[0]
        #for idx, xi in enumerate(sample):
        #    xi = xi.cpu().detach().numpy()
        #    cv.imwrite("{idx}_frame.jpg", xi)
        #exit()
        #print("SheepherdingA2C.forward(): x.size(): " + str(x.size()))
        
        #fx = x.float() / 256
        """
        if os.path.exists("envs"):
            rmtree("envs")
        os.makedirs("envs")

        for env_idx in range(0,x.size()[0]):
            #env_idx = 0
            os.makedirs("envs/env_{}".format(env_idx))
            frames = x[env_idx,:,:,:]
            num_of_frames = int(frames.size()[0]/3)
            for frame_idx in range(0,num_of_frames):
                start_frame_idx = frame_idx*3
                end_frame_idx = (frame_idx+1)*3
                frame = frames[start_frame_idx:end_frame_idx,:,:]
                frame = frame.cpu().detach().numpy()
                frame = frame.swapaxes(0,2)
                cv.imwrite("envs/env_{}/frame_{}.png".format(env_idx,frame_idx), frame)
        """
        fx = x.float() 
        conv_out = self.conv(fx).view(fx.size()[0], -1)        
        #print("conv_out.size(): " + str(conv_out.size()))
        #print("SheepherdingA2C.forward(): conv_out.size(): " + str(conv_out.size()))
        r1 = self.policy(conv_out)
        r2 = self.value(conv_out)
        #print("SheepherdingA2C.forward(): r1.size(): " + str(r1.size()))
        #print("SheepherdingA2C.forward(): r2.size(): " + str(r2.size()))
        #print("r1.size(): " + str(r1.size()))
        #print("r2.size(): " + str(r2.size()))
        #exit()
        return r1, r2 


def unpack_batch(batch, net, device='cpu'):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []                
    
    for idx, exp in enumerate(batch):     
        #print("exp.state: " + str(exp.state))                           
        if isinstance(exp.state,tuple):
            states.append(np.array(exp.state[0], copy=False))
        else:
            states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))
    
    states_v = torch.FloatTensor(np.array(states, copy=False)).to(device)
    actions_t = torch.LongTensor(actions).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(device)
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        last_vals_np *= GAMMA ** REWARD_STEPS
        rewards_np[not_done_idx] += last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)

    return states_v, actions_t, ref_vals_v

def play_game(agent, env, folder_path, game_num):    
    print("play_game")
    #os.makedirs(folder_path)
    state, _ = env.reset()    
    total_reward = 0
    done = False
    while not done:
        action = agent([state])[0][0]        
        state, reward, done, _, _ = env.step(action)
        #print("reward: "+str(reward))
        #print("total_reward: "+str(total_reward))                
        total_reward += reward    
    videofilepath = folder_path + "/sheepherding_{}_{:.3f}.gif".format(game_num, np.round(total_reward,2))    
    env.store_frames_in_gif(videofilepath)
    

if __name__ == "__main__":
    register(
        id="Sheepherding-v0",
        entry_point="sheepherding:Sheepherding",
        max_episode_steps=300,
        reward_threshold=1000000,
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    #make_env = lambda: ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))
    
    #self.env = Game(visualize=False, load_map=True, map_name=self.map_name)
    #env = Sheepherding(**strombom_typical_values)
    #play_env = gym.make("Sheepherding-v0")
    game_storing_folder = "training_games"
    if os.path.exists(game_storing_folder):
        rmtree(game_storing_folder)
    os.makedirs(game_storing_folder)
    #print("game_storing_folder: " +  str(game_storing_folder))
    #print("os.path.exists(game_storing_folder): " + str(os.path.exists(game_storing_folder)))

    #make_env = lambda: Sheepherding(**strombom_typical_values)
    stack_frames = 8
    skip_frames = 4
    make_env = lambda: ptan.common.wrappers.wrap_dqn(gym.make("Sheepherding-v0"), stack_frames=stack_frames, skip_frames=skip_frames)
    play_env = ptan.common.wrappers.wrap_dqn(gym.make("Sheepherding-v0"), stack_frames=stack_frames, skip_frames=skip_frames)
    envs = [make_env() for _ in range(NUM_ENVS)]
    writer = SummaryWriter(comment="-sheepherding-a2c_" + args.name)
    #print("hemlo")
    net = SheepherdingA2C((stack_frames*3,224,224), 9).to(device)
    #print(net)
    
    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
    #agent = ptan.agent.ActorCriticAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
    #agent = ptan.agent.ActorCriticAgent(net, apply_softmax=True, device=device)
    
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS, vectorized=False)
    #exp_source = ptan.experience.ExperienceSourceRollouts(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    batch = []      
    folder_path = str(game_storing_folder)+"/"+str(0)                    
    #play_game(agent, play_env, folder_path)            
    #exit()
    with common.RewardTracker(writer, stop_reward=200) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=BATCH_SIZE) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                batch.append(exp)

                # handle new rewards
                new_rewards = exp_source.pop_total_rewards()
                if new_rewards:
                    if tracker.reward(new_rewards[0], step_idx):
                        break

                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_t, vals_ref_v = unpack_batch(batch, net, device=device)
                batch.clear()

                optimizer.zero_grad()
                logits_v, value_v = net(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                log_prob_v = F.log_softmax(logits_v, dim=1)
                adv_v = vals_ref_v - value_v.detach()
                log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_t]
                loss_policy_v = -log_prob_actions_v.mean()

                prob_v = F.softmax(logits_v, dim=1)
                entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

                # calculate policy gradients only
                loss_policy_v.backward(retain_graph=True)
                grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                        for p in net.parameters()
                                        if p.grad is not None])

                # apply entropy and value gradients
                loss_v = entropy_loss_v + loss_value_v
                loss_v.backward()
                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                optimizer.step()
                # get full loss
                loss_v += loss_policy_v

                tb_tracker.track("advantage",       adv_v, step_idx)
                tb_tracker.track("values",          value_v, step_idx)
                tb_tracker.track("batch_rewards",   vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy",    entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy",     loss_policy_v, step_idx)
                tb_tracker.track("loss_value",      loss_value_v, step_idx)
                tb_tracker.track("loss_total",      loss_v, step_idx)
                tb_tracker.track("grad_l2",         np.sqrt(np.mean(np.square(grads))), step_idx)
                tb_tracker.track("grad_max",        np.max(np.abs(grads)), step_idx)
                tb_tracker.track("grad_var",        np.var(grads), step_idx)
    """
    step_idx = 0
    iters = 0
    with common.RewardTracker(writer, stop_reward=np.inf) as tracker:
       with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for mb_states, mb_rewards, mb_actions, mb_values in exp_source:
                iters += 1
                # handle new rewards
                new_rewards = exp_source.pop_total_rewards()
                if new_rewards:
                    if tracker.reward(np.mean(new_rewards), step_idx):
                        break

                optimizer.zero_grad()
                states_v = torch.FloatTensor(mb_states).to(device)
                mb_adv = mb_rewards - mb_values
                adv_v = torch.FloatTensor(mb_adv).to(device)
                actions_t = torch.LongTensor(mb_actions).to(device)
                vals_ref_v = torch.FloatTensor(mb_rewards).to(device)

                logits_v, value_v = net(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                log_prob_v = F.log_softmax(logits_v, dim=1)
                log_prob_actions_v = adv_v * log_prob_v[range(len(mb_states)), actions_t]
                loss_policy_v = -log_prob_actions_v.mean()

                prob_v = F.softmax(logits_v, dim=1)
                entropy_loss_v = (prob_v * log_prob_v).sum(dim=1).mean()

                # apply entropy and value gradients
                loss_v = loss_policy_v + ENTROPY_BETA * entropy_loss_v + loss_value_v
                loss_v.backward()
                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                optimizer.step()

                tb_tracker.track("advantage",       adv_v, step_idx)
                tb_tracker.track("values",          value_v, step_idx)
                tb_tracker.track("batch_rewards",   vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy",    entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy",     loss_policy_v, step_idx)
                tb_tracker.track("loss_value",      loss_value_v, step_idx)
                tb_tracker.track("loss_total",      loss_v, step_idx)
                step_idx += NUM_ENVS * REWARD_STEPS
                #if iters % 50 == 0:
                #    play_game(agent, play_env, game_storing_folder, iters)
    """