import math

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torchvision.models import resnet18
import gym
from retro_contest.local import make
from tqdm import tqdm

from sonic_util import get_sonic_specific_actions

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        #actor
        self.action_cnn = nn.Sequential(*list(resnet18(pretrained=False).children())[:-1])
        self.action_linear = nn.Linear(512, action_dim)
        
        # critic
        self.value_cnn = nn.Sequential(*list(resnet18(pretrained=False).children())[:-1])
        self.value_linear = nn.Linear(512, 1)
        
        self.eval_mode()
        
    def forward_actor(self, state):
        net_features = self.action_cnn(state).squeeze()
        action_values = self.action_linear(net_features)
        action_probs = nn.functional.softmax(action_values, dim=-1)
        return action_probs
    
    def forward_critic(self, state):
        net_features = self.value_cnn(state).squeeze()
        state_value = self.value_linear(net_features)
        return state_value
        
    def act(self, state, memory):
        state = torch.from_numpy(state).float().permute(2, 0, 1).unsqueeze(dim=0).to(device)
        action_probs = self.forward_actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        memory.states.append(state.detach().to('cpu'))
        memory.actions.append(action.detach().to('cpu'))
        memory.logprobs.append(dist.log_prob(action).detach().to('cpu'))
        
        return action.item()
    
    def evaluate(self, state, action):
        action_probs = self.forward_actor(state.squeeze())
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.forward_critic(state.squeeze())
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
    
    def train_mode(self):
        self.action_cnn.train()
        self.value_cnn.train()
    
    def eval_mode(self):
        self.action_cnn.eval()
        self.value_cnn.eval()
        
class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, net_batch_size):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.net_batch_size = net_batch_size
        
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def update(self, memory):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.stack(memory.states)
        old_actions = torch.stack(memory.actions)
        old_logprobs = torch.stack(memory.logprobs)
        
        self.policy.train_mode()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            epoch_indices = torch.arange(old_states.shape[0])
            for i in range(math.ceil(old_states.shape[0] / self.net_batch_size)):
                batch_indices = epoch_indices[i * self.net_batch_size : min((i + 1) * self.net_batch_size, old_states.shape[0])]
                old_states_batch = old_states[batch_indices].to(device).detach()
                old_actions_batch = old_actions[batch_indices].to(device).detach()
                old_logprobs_batch = old_logprobs[batch_indices].to(device).detach()
                rewards_batch = rewards[batch_indices].to(device)

                # Evaluating old actions and values :
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states_batch, old_actions_batch)
                
                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - old_logprobs_batch.detach())
                    
                # Finding Surrogate Loss:
                advantages = rewards_batch - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards_batch) - 0.01 * dist_entropy
                 
                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.policy.eval_mode()
        
def main():
    ############## Hyperparameters ##############
    env = make(game='SonicTheHedgehog-Genesis', state='SpringYardZone.Act3')
    state_dim = env.observation_space.shape[0]
    action_dim = 8
    sonic_actions = get_sonic_specific_actions()
    render = False
    solved_reward = 4e3         # stop training if avg_reward > solved_reward
    log_interval = 5           # print avg reward in the interval
    max_episodes = 50000        # max training episodes
    max_timesteps = 4500         # max timesteps in one episode
    n_latent_var = 64           # number of variables in hidden layer
    update_timestep = 500      # update policy every n timesteps
    lr = 0.002
    batch_size = 200
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 1                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None
    #############################################
    
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, batch_size)
    print(lr,betas)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    
    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            timestep += 1
            
            # Running policy_old:
            action = ppo.policy_old.act(state, memory)
            state, reward, done, _ = env.step(sonic_actions[action])
            
            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0
            
            running_reward += reward
            if render:
                env.render()
            if done:
                break
                
        avg_length += t
        
        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
            break
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
            
if __name__ == '__main__':
    main()
