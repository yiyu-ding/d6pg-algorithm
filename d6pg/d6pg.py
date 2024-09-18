import torch
import torch.nn.functional as F
import numpy as np
import collections
import matplotlib.pyplot as plt
import random
import tools

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
        # print(x)
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
        action = self.actor(state)

        if explore:
            action = gumbel_softmax(action)
        else:
            action = onehot_from_logits(action)

        return action.detach().cpu().numpy()[0]

    def take_action_test(self, state, explore=False):
        action = self.actor(state)
        # print("state", end=' ')
        # print(state)
        # print("action1", end=' ')
        # print(action)
        if explore:
            action = gumbel_softmax(action)
        else:
            action = onehot_from_logits(action)
        # print("action2", end=' ')
        # print(action)
        return action.detach().cpu().numpy()[0]

    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) +
                                    param.data * tau)

    def consensus_update(self, this_net, neighbor_net, ratio):
        # for param_1, param_2 in zip(net_1.parameters(), net_2.parameters()):
        #     param_sum.data.copy_(param_1.data + param_2.data)
        neighbor_net_temp = neighbor_net

        for i in range(len(neighbor_net_temp) - 1):
            for param_i, param_i_next in zip(neighbor_net_temp[i].parameters(), neighbor_net_temp[i + 1].parameters()):
                param_i_next.data.copy_(param_i.data + param_i_next.data)

        for param_this, param_last in zip(this_net.parameters(), neighbor_net_temp[-1].parameters()):
            param_this.data.copy_(param_this.data - ratio * (param_this.data - param_last.data))

    # def consensus_update_test(self, this_net, neighbor_net, ratio):
    #     # for param_1, param_2 in zip(net_1.parameters(), net_2.parameters()):
    #     #     param_sum.data.copy_(param_1.data + param_2.data)
    #     neighbor_net_temp = neighbor_net
    #
    #     print("parameters", end = ' ')
    #     print(this_net.parameters())
    #
    #     for i in range(len(neighbor_net_temp) - 1):
    #         for param_i, param_i_next in zip(neighbor_net_temp[i].parameters(), neighbor_net_temp[i + 1].parameters()):
    #             param_i_next.data.copy_(param_i.data + param_i_next.data)
    #
    #     for param_this, param_last in zip(this_net.parameters(), neighbor_net_temp[-1].parameters()):
    #         param_this.data.copy_(param_this.data - ratio * (param_this.data - param_last.data))

    #
    # def consensus_update_test(self, this_net, neighbor_net, ratio):
    #     # for param_1, param_2 in zip(net_1.parameters(), net_2.parameters()):
    #     #     param_sum.data.copy_(param_1.data + param_2.data)
    #     neighbor_net_temp = neighbor_net
    #
    #     num = len(neighbor_net)
    #
    #     print("parameters", end = ' ')
    #     # print(this_net.parameters())
    #
    #     for para in this_net.parameters():
    #         print(para.data, end = ' ')
    #
    #     print("==============================================")
    #
    #     for i in range(len(neighbor_net_temp) - 1):
    #         for param_i, param_i_next in zip(neighbor_net_temp[i].parameters(), neighbor_net_temp[i + 1].parameters()):
    #             param_i_next.data.copy_(param_i.data + param_i_next.data)
    #
    #     for param_this, param_last in zip(this_net.parameters(), neighbor_net_temp[-1].parameters()):
    #         param_this.data.copy_(param_this.data - ratio * (param_this.data - (param_last.data * (1.0/num))))

    def consensus_update_test(self, this_net, neighbor_net, ratio):
        # for param_1, param_2 in zip(net_1.parameters(), net_2.parameters()):
        #     param_sum.data.copy_(param_1.data + param_2.data)
        neighbor_net_temp = neighbor_net

        num = len(neighbor_net)

        # print("parameters", end = ' ')
        # print(this_net.parameters())

        # for para in this_net.parameters():
        #     print(para.data, end = ' ')

        # print(next(this_net.parameters()), end = ' ')
        #
        # print("==============================================")

        worker_state_dict = [x.state_dict() for x in neighbor_net_temp]
        primary_state_dict = this_net.state_dict()
        weight_keys = list(worker_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for i in range(len(neighbor_net_temp)):
                key_sum = key_sum + worker_state_dict[i][key]
            fed_state_dict[key] = (key_sum / len(neighbor_net_temp)) * ratio + (primary_state_dict[key]) * (1.0 - ratio)
        #### update fed weights to fl model
        this_net.load_state_dict(fed_state_dict)



        # for param_this, param_last in zip(this_net.parameters(), neighbor_net_temp[-1].parameters()):
        #     param_this.data = param_this.data - ratio * (param_this.data - (param_last.data * (1.0/num)))


class DGDDTDEMADDPG:
    def __init__(self, env, device, actor_lr, critic_lr, hidden_dim,
                 state_dims, action_dims, critic_input_dims, gamma, tau):
        self.agents = []
        for i in range(len(env.agents)):
            self.agents.append(
                DDPG(state_dims[i], action_dims[i], critic_input_dims[i],
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
        return [
            agent.take_action(state, explore)
            for agent, state in zip(self.agents, states)
        ]

    def take_action_test(self, states, explore):
        states = [
            torch.tensor([states[i]], dtype=torch.float, device=self.device)
            for i in range(len(env.agents))
        ]
        return [
            agent.take_action_test(state, explore)
            for agent, state in zip(self.agents, states)
        ]

    def take_action_adj(self, states_adj, explore):

        states = []

        for i in range(len(self.agents)):
            states.append(states_adj[i][0])

        states = [
            torch.tensor([states[i]], dtype=torch.float, device=self.device)
            for i in range(len(env.agents))
        ]
        return [
            agent.take_action(state, explore)
            for agent, state in zip(self.agents, states)
        ]

    # def take_action_adj(self, obs_adj, explore):
    #     action_adj = []
    #     for agent, obs in zip(self.agents, obs_adj):
    #         action_adj.append(agent.take_action_adj(obs, explore))

    def update(self, sample):

        obs = [[] for _ in range(len(env.agents))]
        act = [[] for _ in range(len(env.agents))]
        rew = [[] for _ in range(len(env.agents))]
        next_obs = [[] for _ in range(len(env.agents))]
        done = [[] for _ in range(len(env.agents))]

        for i in range(len(env.agents)):
            obs[i], act[i], rew[i], next_obs[i], done[i] = sample[i]

        for i in range(len(env.agents)):
            self.agents[i].critic_optimizer.zero_grad()

        all_target_act = [[] for _ in range(len(env.agents))]

        for i in range(len(env.agents)):
            all_target_act[i] = [
                onehot_from_logits(pi(_next_obs))
                for pi, _next_obs in zip(self.target_policies, next_obs[i])
            ]

        target_critic_input = [torch.cat((*(next_obs[i]), *(all_target_act[i])), dim=1) for i in range(len(env.agents))]

        target_critic_value = [rew[i][0].view(-1, 1) + self.gamma * self.agents[i].target_critic(target_critic_input[i]) * (1 - done[i][0].view(-1, 1)) for i in range(len(env.agents))]

        # target_critic_value[i] = rew[i][0].view(-1, 1) + self.trust * (sum(rew[i]).view(-1, 1) - rew[i][0].view(-1, 1)) \
        #                          + self.gamma * self.agents[i].target_critic(target_critic_input[i]) * (1 - done[i].view(-1, 1))

        # print(len(obs), len(act))
        #
        # print(len(obs[0]), len(act[0]))
        #
        # print(obs[0][0].shape, act[0][0].shape)
        #
        # print('division')

        critic_input = [torch.cat((*obs[i], *act[i]), dim=1) for i in range(len(env.agents))]

        # print(critic_input[0])
        # # exit()
        #
        # print(self.agents[0])
        #
        # print(self.agents[0].critic_input_dim)
        #
        # print(critic_input[0].shape)
        #
        # exit()

        # test = (self.agents[0]).critic(critic_input[0])

        # print(test)

        # print("critic_input", end = ' ')
        # print(critic_input)

        critic_value = [self.agents[i].critic(critic_input[i]) for i in range(len(env.agents))]

        # print("critic_value", end = ' ')
        # print(critic_value)

        critic_loss = [self.critic_criterion(critic_value[i], target_critic_value[i].detach()) for i in range(len(env.agents))]

        for i in range(len(env.agents)):
            critic_loss[i].backward()
            self.agents[i].critic_optimizer.step()
            self.agents[i].actor_optimizer.zero_grad()

        cur_actor_out = [self.agents[i].actor(obs[i][0]) for i in range(len(env.agents))]

        cur_act_vf_in = [gumbel_softmax(cur_actor_out[i]) for i in range(len(env.agents))]

        adj_actor_acs = [[] for _ in range(len(env.agents))]

        for i in range(len(env.agents)):
            for j, (pi, _obs) in enumerate(zip(self.policies, obs[i])):
                if j == i:
                    adj_actor_acs[i].append(cur_act_vf_in[i])
                else:
                    adj_actor_acs[i].append(onehot_from_logits(pi(_obs)))

        vf_in = [torch.cat((*obs[i], *adj_actor_acs[i]), dim=1) for i in range(len(env.agents))]

        actor_loss = [-self.agents[i].critic(vf_in[i]).mean() for i in range(len(env.agents))]

        for i in range(len(env.agents)):
            actor_loss[i] += (cur_actor_out[i]**2).mean() * 1e-3
            actor_loss[i].backward()
            self.agents[i].actor_optimizer.step()

    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)

    def distributed_optimization(self):
        for i, agt in enumerate(self.agents):

            cur_agt_neighbor_net = []

            for j in range(len(self.agents)):
                if env.adj_matrix[i][j] == 1:
                    cur_agt_neighbor_net.append(self.agents[j].critic)

            agt.consensus_update(agt.critic, cur_agt_neighbor_net, 0.03)


    def distributed_optimization_test(self):
        for i, agt in enumerate(self.agents):

            cur_agt_neighbor_net = []

            for j in range(len(self.agents)):
                if env.adj_matrix[i][j] == 1:
                    cur_agt_neighbor_net.append(self.agents[j].critic)

            agt.consensus_update_test(agt.critic, cur_agt_neighbor_net, 0.03)

            # print("", end='\n')

def evaluate(env_id, dgddtdemaddpg, n_episode=10, episode_length=40):
    env = make_env(env_id)
    returns = np.zeros(len(env.agents))
    for _ in range(n_episode):
        obs = env.reset()
        for t_i in range(episode_length):
            actions = dgddtdemaddpg.take_action(obs, explore=False)
            obs, rew, done, info = env.step(actions)
            rew = np.array(rew)
            returns += rew / n_episode
    returnavg = sum(returns) / len(env.agents)
    # return returns.tolist()
    return returnavg


def demonstrate(env_id, dgddtdemaddpg, n_episode=1, episode_length=40):
    # print("demonstrate")
    env = make_env(env_id)
    returns = np.zeros(len(env.agents))
    for _ in range(n_episode):
        obs = env.reset()
        # env.showjudge = True
        # for marks in env.world.landmarks:
        #     print(marks.state.p_pos)

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

        # print("division")
        for t_i in range(episode_length):
            actions = dgddtdemaddpg.take_action_test(obs, explore=False)
            # print("action", end=' ')
            # print(actions)
            obs, rew, done, info = env.step_show(actions)
            # print("obs:")
            # print(obs)
            rew = np.array(rew)
            returns += rew / n_episode
        env.showjudge = False
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

env = make_env(env_id)

def runoneturn():

    replay_buffer = [rl_utils.ReplayBuffer(buffer_size) for _ in range(len(env.agents))]

    state_dims = []
    action_dims = []

    for action_space in env.action_space:
        action_dims.append(action_space.n)
    for state_space in env.observation_space:
        state_dims.append(state_space.shape[0])

    critic_input_dims = [0] * len(env.agents)

    #print(action_dims)
    #print(state_dims)

    for i in range(env.n):
        for j in range(len(env.agents)):
            if env.adj_matrix[i][j] == 2 or env.adj_matrix[i][j] == 1:
                critic_input_dims[i] += (state_dims[j] + action_dims[j])
            else:
                pass

    #print(critic_input_dims)

    dgddtdemaddpg = DGDDTDEMADDPG(env, device, actor_lr, critic_lr, hidden_dim, state_dims,
                    action_dims, critic_input_dims, gamma, tau)


    return_list = []
    total_step = 0

    state_adj = [[] for _ in range(len(env.agents))]

    actions = [[] for _ in range(len(env.agents))]

    reward_n_adj = [[] for _ in range(len(env.agents))]
    next_state_adj = [[] for _ in range(len(env.agents))]
    done_n_adj = [[] for _ in range(len(env.agents))]
    info_adj = [[] for _ in range(len(env.agents))]

    sample = [[] for _ in range(len(env.agents))]

    #print(len(env.agents))
    #print(env.n)

    for i_episode in range(num_episodes):
        state_adj = env.reset_adj()
        # ep_returns = np.zeros(len(env.agents))

        # print(state_adj)

        for e_i in range(episode_length):

            actions = dgddtdemaddpg.take_action_adj(state_adj, explore=True)

            # print('actions')

            # print(actions)

            # exit()

            actions_adj = [[] for _ in range(len(env.agents))]

            for i in range(env.n):
                for j in range(len(env.agents)):
                    if env.adj_matrix[i][j] == 2:
                        actions_adj[i].append(actions[j])
                    if env.adj_matrix[i][j] == 1:
                        actions_adj[i].append(actions[j])
                    else:
                        pass

            # print('actions_adj')

            # exit()

            # print(actions_adj)

            next_state_adj, reward_n_adj, done_n_adj, info_adj = env.step_adj(actions)

            # print(actions_adj)
            # print(next_state_adj)
            # print(reward_n_adj)
            # print(done_n_adj)
            # print(info_adj)

            # exit()

            for i in range(env.n):
                replay_buffer[i].add(state_adj[i], actions_adj[i], reward_n_adj[i], next_state_adj[i], done_n_adj[i])
                state_adj[i] = next_state_adj[i]

            # print(len(replay_buffer[0].buffer[total_step][0]))
            # print(len(replay_buffer[0].buffer[total_step][1]))
            # print(len(replay_buffer[0].buffer[total_step][2]))
            # print(len(replay_buffer[0].buffer[total_step][3]))
            # print(len(replay_buffer[0].buffer[total_step][4]))

            total_step += 1

            # print(replay_buffer[0].buffer)

            # exit()

            # print(total_step)
            # print(replay_buffer)
            # print(sample)

            # print(replay_buffer[0])

            judge = 0

            for i in range(len(env.agents)):
                if replay_buffer[i].size() >= minimal_size:
                    judge += 1

            if (judge == len(env.agents)) and (total_step % update_interval == 0):

                def stack_array(x):
                    rearranged = [[sub_x[i] for sub_x in x]
                                  for i in range(len(x[0]))]
                    return [
                        torch.FloatTensor(np.vstack(aa)).to(device)
                        for aa in rearranged
                    ]

                for i in range(env.n):

                    sample[i] = replay_buffer[i].sample(batch_size)

                    sample[i] = [stack_array(x) for x in sample[i]]

                # print('replaybuffer')
                #
                # print(len(replay_buffer[0].buffer))
                # print(len(replay_buffer[0].buffer[467][0]))
                # print(len(replay_buffer[0].buffer[467][1]))
                # print(len(replay_buffer[0].buffer[467][2]))
                # print(len(replay_buffer[0].buffer[467][3]))
                # print(len(replay_buffer[0].buffer[467][4]))

                # print('sample')
                #
                # print(len(sample))
                # print(len(sample[0]))
                # print(len(sample[0][0][0]))g
                # print(len(sample[0][1]))
                # print(len(sample[0][2]))
                # print(len(sample[0][3]))
                # print(len(sample[0][4]))
                #
                # exit()
                # dgddtdemaddpg.distributed_optimization()

                dgddtdemaddpg.distributed_optimization_test()

                dgddtdemaddpg.update(sample)

                dgddtdemaddpg.update_all_targets()

        if (i_episode + 1) % 100 == 0:
            ep_returns = evaluate(env_id, dgddtdemaddpg, n_episode=100)
            return_list.append(ep_returns)
            print(f"Episode: {i_episode+1}, {ep_returns}")
            #####################################################################
            # if (i_episode + 1) / 100 == 1:
            #     #env.showjudge = True
            #     demonstrate(env_id, dgddtdemaddpg, n_episode=1)
            #     #env.showjudge = False
            if (i_episode + 1) / 100 == 50:
                demonstrate(env_id, dgddtdemaddpg, n_episode=1)

    return_array = np.array(return_list)

    return return_array

# return_array = np.array(return_list)
# for i, agent_name in enumerate(["agent_0"]):
#     plt.figure()
#     plt.plot(
#         np.arange(return_array.shape[0]) * 100,
#         rl_utils.moving_average(return_array[:, i], 9))
#     plt.xlabel("Episodes")
#     plt.ylabel("Returns")
#     plt.title("Agent0/1/2/3/4 's Rewards by DTDE-MADDPG(DGD added)")
#     plt.show()

if __name__ == '__main__':

    turn_num = 1

    turn_return = []

    for i in range(turn_num):
        turn_return.append(runoneturn())

    turn_array = np.array(turn_return)

    print('=================== turn end ===================')

    print(turn_array)

    print('=================== division ===================')

    # turn_array

    # if env_id == "simple_air_local":
    #     np.savetxt('./result/uav-8/local_d6pg.csv', turn_array, fmt='%f', delimiter=',')
    # if env_id == "simple_air_global":
    #     np.savetxt('./result/uav-8/global_d6pg.csv', turn_array, fmt='%f', delimiter=',')
