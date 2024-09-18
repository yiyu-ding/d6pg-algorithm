import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import tools

# torch.set_printoptions(threshold=np.inf)

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append("experimental-envs")

import gym
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

import netopu

topuname = netopu.Net_Topo_Loop_5u

hidden_dim = 128

env_id = "simple_spread"

def make_env(scenario_name):
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, topuname, scenario.reset_world, scenario.reward,
                        scenario.observation)
    return env

def onehot_from_logits(logits, eps=0.01):
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[
        np.random.choice(range(logits.shape[1]), size=logits.shape[0])
    ]],
                                       requires_grad=False).to(logits.device)
    return torch.stack([
        argmax_acs[i] if r > eps else rand_acs[i]
        for i, r in enumerate(torch.rand(logits.shape[0]))
    ])


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    U = torch.autograd.Variable(tens_type(*shape).uniform_(),
                                requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(
        logits.device)
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0):
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y)
    y = (y_hard.to(logits.device) - y).detach() + y
    return y

class TwoLayerFC(torch.nn.Module):
    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DDPG:
    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim,
                 actor_lr, critic_lr, device):
        self.actor = TwoLayerFC(state_dim, action_dim, hidden_dim).to(device)
        self.target_actor = TwoLayerFC(state_dim, action_dim,
                                       hidden_dim).to(device)
        self.critic = TwoLayerFC(critic_input_dim, 1, hidden_dim).to(device)
        self.target_critic = TwoLayerFC(critic_input_dim, 1,
                                        hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.critic_input_dim = critic_input_dim

    def take_action(self, state, explore=False):

        #print(state)

        #print(self.actor)

        action = self.actor(state)

        if explore:
            action = gumbel_softmax(action)
        else:
            action = onehot_from_logits(action)
        return action.detach().cpu().numpy()[0]

    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) +
                                    param.data * tau)

class MADDPG:
    def __init__(self, env, device, actor_lr, critic_lr, hidden_dim,
                 state_dims, action_dims, critic_input_dim, gamma, tau):
        self.agents = []
        for i in range(len(env.agents)):
            self.agents.append(
                DDPG(state_dims[i], action_dims[i], critic_input_dim,
                     hidden_dim, actor_lr, critic_lr, device))
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device

    @property
    def policies(self):
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):
        return [agt.target_actor for agt in self.agents]

    def take_action(self, states, explore):
        states = [
            torch.tensor([states[i]], dtype=torch.float, device=self.device)
            for i in range(len(env.agents))
        ]

        #print(states)
        #exit()

        return [
            agent.take_action(state, explore)
            for agent, state in zip(self.agents, states)
        ]

    def update(self, sample, i_agent):
        obs, act, rew, next_obs, done = sample
        cur_agent = self.agents[i_agent]

        cur_agent.critic_optimizer.zero_grad()
        all_target_act = [
            onehot_from_logits(pi(_next_obs))
            for pi, _next_obs in zip(self.target_policies, next_obs)
        ]
        target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1)
        target_critic_value = rew[i_agent].view(
            -1, 1) + self.gamma * cur_agent.target_critic(
                target_critic_input) * (1 - done[i_agent].view(-1, 1))

        #print(len(obs), len(act))

        #print(obs[0].shape, act[0].shape)

        #print('obs')

        #print(obs)

        #print('*obs')

        #print(*obs)

        #print('act')

        #print(act)

        #print('*act')

        #print(*act)

        critic_input = torch.cat((*obs, *act), dim=1)

        #print(critic_input)

        # exit()

        #print(cur_agent.critic_input_dim)

        #print(critic_input.shape)

        # print("critic_input", end = ' ')
        # print(critic_input)

        critic_value = cur_agent.critic(critic_input)

        # print("critic_value", end = ' ')
        # print(critic_value)

        #exit()

        #print(critic_value)

        # exit()

        critic_loss = self.critic_criterion(critic_value,
                                            target_critic_value.detach())
        critic_loss.backward()
        cur_agent.critic_optimizer.step()

        cur_agent.actor_optimizer.zero_grad()
        cur_actor_out = cur_agent.actor(obs[i_agent])
        cur_act_vf_in = gumbel_softmax(cur_actor_out)
        all_actor_acs = []
        for i, (pi, _obs) in enumerate(zip(self.policies, obs)):
            if i == i_agent:
                all_actor_acs.append(cur_act_vf_in)
            else:
                all_actor_acs.append(onehot_from_logits(pi(_obs)))
        vf_in = torch.cat((*obs, *all_actor_acs), dim=1)
        actor_loss = -cur_agent.critic(vf_in).mean()
        actor_loss += (cur_actor_out**2).mean() * 1e-3
        actor_loss.backward()
        cur_agent.actor_optimizer.step()

    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)


def evaluate(env_id, maddpg, n_episode=10, episode_length=40):
    env = make_env(env_id)
    returns = np.zeros(len(env.agents))
    for _ in range(n_episode):
        obs = env.reset()
        for t_i in range(episode_length):
            actions = maddpg.take_action(obs, explore=False)
            obs, rew, done, info = env.step(actions)
            rew = np.array(rew)
            returns += rew / n_episode
    returnavg = sum(returns) / len(env.agents)
    #return returns.tolist()
    return returnavg


def demonstrate(env_id, maddpg, n_episode=1, episode_length=40):
    # print("demonstrate")
    env = make_env(env_id)
    returns = np.zeros(len(env.agents))
    for _ in range(n_episode):
        obs = env.reset()
        #env.showjudge = True
        # for marks in env.world.landmarks:
        #     print(marks.state.p_pos)

        # env.world.landmarks[0].state.p_pos = np.array([0.5, 1.0])
        # env.world.landmarks[1].state.p_pos = np.array([0.5, 1.5])
        # env.world.landmarks[2].state.p_pos = np.array([1.0, 0.5])
        # env.world.landmarks[3].state.p_pos = np.array([1.0, 1.0])
        # env.world.landmarks[4].state.p_pos = np.array([1.5, 0.5])

        # env.agents[0].state.p_pos = np.array([-0.5, -1.0])
        # env.agents[1].state.p_pos = np.array([-0.5, -1.5])
        # env.agents[2].state.p_pos = np.array([-1.0, -0.5])
        # env.agents[3].state.p_pos = np.array([-1.0, -1.0])
        # env.agents[4].state.p_pos = np.array([-1.5, -0.5])

        for t_i in range(episode_length):
            actions = maddpg.take_action(obs, explore=False)
            obs, rew, done, info = env.step_show(actions)
            #print(obs)
            rew = np.array(rew)
            returns += rew / n_episode
        #env.showjudge = False
    returnavg = sum(returns) / len(env.agents)
    #return returns.tolist()

    return returnavg



num_episodes = 6000
episode_length = 40 
buffer_size = 200000
actor_lr = 1e-2
critic_lr = 1e-2
gamma = 0.95
tau = 1e-2
batch_size = 2048
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
update_interval = 100
minimal_size = 8000
# trust = 0.2

# env_id = "simple_spread_v3"
env = make_env(env_id)

print(len(env.agents))

def runoneturn():

    replay_buffer = rl_utils.ReplayBuffer(buffer_size)

    state_dims = []
    action_dims = []
    for action_space in env.action_space:
        action_dims.append(action_space.n)
    for state_space in env.observation_space:
        state_dims.append(state_space.shape[0])
    critic_input_dim = sum(state_dims) + sum(action_dims)


    # print(state_dims)
    # print(action_dims)
    # print(critic_input_dim)

    maddpg = MADDPG(env, device, actor_lr, critic_lr, hidden_dim, state_dims,
                    action_dims, critic_input_dim, gamma, tau)


    return_list = []
    total_step = 0
    for i_episode in range(num_episodes):
        state = env.reset()
        # ep_returns = np.zeros(len(env.agents))
        # print(state)
        for e_i in range(episode_length):
            actions = maddpg.take_action(state, explore=True)

            next_state, reward, done, _ = env.step(actions)

            replay_buffer.add(state, actions, reward, next_state, done)
            state = next_state

            total_step += 1
            if replay_buffer.size(
            ) >= minimal_size and total_step % update_interval == 0:
                sample = replay_buffer.sample(batch_size)

                def stack_array(x):
                    rearranged = [[sub_x[i] for sub_x in x]
                                  for i in range(len(x[0]))]
                    return [
                        torch.FloatTensor(np.vstack(aa)).to(device)
                        for aa in rearranged
                    ]

                sample = [stack_array(x) for x in sample]
                for a_i in range(len(env.agents)):
                    maddpg.update(sample, a_i)
                maddpg.update_all_targets()
        if (i_episode + 1) % 100 == 0:
            ep_returns = evaluate(env_id, maddpg, n_episode=100)
            return_list.append(ep_returns)
            print(f"Episode: {i_episode+1}, {ep_returns}")
            #####################################################################
            # if (i_episode + 1) / 100 == 50:
            #     #env.showjudge = True
            #     demonstrate(env_id, maddpg, n_episode=1)
            #     #env.showjudge = False
            if (i_episode + 1) / 100 == 50:
                demonstrate(env_id, maddpg, n_episode=1)

    return_array = np.array(return_list)

    return return_array

# for i, agent_name in enumerate(["agent_0"]):
#     plt.figure()
#     plt.plot(
#         np.arange(return_array.shape[0]) * 100,
#         rl_utils.moving_average(return_array[:, i], 9))
#     plt.xlabel("Episodes")
#     plt.ylabel("Returns")
#     plt.title("Agent0/1/2/3/4 's Rewards by MADDPG")
#     plt.show()

if __name__ == '__main__':

    print(torch.cuda.is_available())

    # if env_id == "simple_spread_v3":
    #     reward = "local"
    # if env_id == "simple_spread":
    #     reward = "global"

    turn_num = 1

    turn_return = []

    for i in range(turn_num):
        print('=================== one new turn ===================')
        turn_return.append(runoneturn())

    turn_array = np.array(turn_return)

    print('=================== turn end ===================')

    print(turn_array)

    # if env_id == "simple_air_local":
    #     np.savetxt('./result/uav-8/local_maddpg.csv', turn_array, fmt='%f', delimiter=',')
    # if env_id == "simple_air_global":
    #     np.savetxt('./result/uav-8/global_maddpg.csv', turn_array, fmt='%f', delimiter=',')
