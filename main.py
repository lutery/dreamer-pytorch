'''
在 Dreamer 算法中，"信念"（belief）是一个关键概念，它表示智能体对当前环境状态的估计。信念状态是通过递归神经网络（如 GRU 或 LSTM）来维护的，并结合过去的观察和动作信息，提供对当前环境状态的估计。

详细解释
信念状态（Belief State）：

信念状态是一个高维向量，用于表示智能体对当前环境状态的估计。
它通过递归神经网络（如 GRU 或 LSTM）来维护，并结合过去的观察和动作信息，提供对当前环境状态的估计。
隐状态（Latent State）：

隐状态是另一个高维向量，用于表示环境的潜在状态。
隐状态通过模型的状态转移网络（Transition Model）来更新，并结合当前的信念状态和动作，预测下一个隐状态。
'''

import argparse
import os

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from env import CONTROL_SUITE_ENVS, GYM_ENVS, Env, EnvBatcher
from memory import ExperienceReplay
from models import ActorModel, Encoder, ObservationModel, RewardModel, TransitionModel, ValueModel, bottle
from planner import MPCPlanner
from utils import FreezeParameters, imagine_ahead, lambda_return, lineplot, write_video

# Hyperparameters
parser = argparse.ArgumentParser(description='PlaNet or Dreamer')
parser.add_argument('--algo', type=str, default='dreamer', help='planet or dreamer')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument(
    '--env',
    type=str,
    default='Pendulum-v0',
    choices=GYM_ENVS + CONTROL_SUITE_ENVS,
    help='Gym/Control Suite environment',
)
parser.add_argument('--symbolic-env', action='store_true', help='Symbolic features')
parser.add_argument('--max-episode-length', type=int, default=1000, metavar='T', help='Max episode length')
parser.add_argument(
    '--experience-size', type=int, default=1000000, metavar='D', help='Experience replay size'
)  # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
parser.add_argument(
    '--cnn-activation-function',
    type=str,
    default='relu',
    choices=dir(F),
    help='Model activation function for a convolution layer',
)
parser.add_argument(
    '--dense-activation-function',
    type=str,
    default='elu',
    choices=dir(F),
    help='Model activation function a dense layer',
)
parser.add_argument(
    '--embedding-size', type=int, default=1024, metavar='E', help='Observation embedding size'
)  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
parser.add_argument('--hidden-size', type=int, default=200, metavar='H', help='Hidden size')
parser.add_argument('--belief-size', type=int, default=200, metavar='H', help='Belief/hidden size')
parser.add_argument('--state-size', type=int, default=30, metavar='Z', help='State/latent size')
parser.add_argument('--action-repeat', type=int, default=2, metavar='R', help='Action repeat')
parser.add_argument('--action-noise', type=float, default=0.3, metavar='ε', help='Action noise')
parser.add_argument('--episodes', type=int, default=1000, metavar='E', help='Total number of episodes')
parser.add_argument('--seed-episodes', type=int, default=5, metavar='S', help='Seed episodes')
parser.add_argument('--collect-interval', type=int, default=100, metavar='C', help='Collect interval')
# 采集数据时要同时采集多少个环境数据
parser.add_argument('--batch-size', type=int, default=50, metavar='B', help='Batch size')
# 每个环境序列采集的长度
parser.add_argument('--chunk-size', type=int, default=50, metavar='L', help='Chunk size')
parser.add_argument(
    '--worldmodel-LogProbLoss',
    action='store_true',
    help='use LogProb loss for observation_model and reward_model training',
)
parser.add_argument(
    '--overshooting-distance',
    type=int,
    default=50,
    metavar='D',
    help='Latent overshooting distance/latent overshooting weight for t = 1',
)
parser.add_argument(
    '--overshooting-kl-beta',
    type=float,
    default=0,
    metavar='β>1',
    help='Latent overshooting KL weight for t > 1 (0 to disable)',
)
parser.add_argument(
    '--overshooting-reward-scale',
    type=float,
    default=0,
    metavar='R>1',
    help='Latent overshooting reward prediction weight for t > 1 (0 to disable)',
)
parser.add_argument('--global-kl-beta', type=float, default=0, metavar='βg', help='Global KL weight (0 to disable)')
parser.add_argument('--free-nats', type=float, default=3, metavar='F', help='Free nats')
parser.add_argument('--bit-depth', type=int, default=5, metavar='B', help='Image bit depth (quantisation)')
parser.add_argument('--model_learning-rate', type=float, default=1e-3, metavar='α', help='Learning rate')
parser.add_argument('--actor_learning-rate', type=float, default=8e-5, metavar='α', help='Learning rate')
parser.add_argument('--value_learning-rate', type=float, default=8e-5, metavar='α', help='Learning rate')
parser.add_argument(
    '--learning-rate-schedule',
    type=int,
    default=0,
    metavar='αS',
    help='Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)',
)
parser.add_argument('--adam-epsilon', type=float, default=1e-7, metavar='ε', help='Adam optimizer epsilon value')
# Note that original has a linear learning rate decay, but it seems unlikely that this makes a significant difference
parser.add_argument('--grad-clip-norm', type=float, default=100.0, metavar='C', help='Gradient clipping norm')
parser.add_argument('--planning-horizon', type=int, default=15, metavar='H', help='Planning horizon distance')
parser.add_argument('--discount', type=float, default=0.99, metavar='H', help='Planning horizon distance')
parser.add_argument('--disclam', type=float, default=0.95, metavar='H', help='discount rate to compute return')
parser.add_argument('--optimisation-iters', type=int, default=10, metavar='I', help='Planning optimisation iterations')
parser.add_argument('--candidates', type=int, default=1000, metavar='J', help='Candidate samples per iteration')
parser.add_argument('--top-candidates', type=int, default=100, metavar='K', help='Number of top candidates to fit')
parser.add_argument('--test', action='store_true', help='Test only')
parser.add_argument('--test-interval', type=int, default=25, metavar='I', help='Test interval (episodes)')
parser.add_argument('--test-episodes', type=int, default=10, metavar='E', help='Number of test episodes')
parser.add_argument('--checkpoint-interval', type=int, default=50, metavar='I', help='Checkpoint interval (episodes)')
parser.add_argument('--checkpoint-experience', action='store_true', help='Checkpoint experience replay')
parser.add_argument('--models', type=str, default='', metavar='M', help='Load model checkpoint')
parser.add_argument('--experience-replay', type=str, default='', metavar='ER', help='Load experience replay')
parser.add_argument('--render', action='store_true', help='Render environment')
args = parser.parse_args()
# todo overshooting_distance的作用时什么
args.overshooting_distance = min(
    args.chunk_size, args.overshooting_distance
)  # Overshooting distance cannot be greater than chunk size
print(' ' * 26 + 'Options')
# 打印参数
for k, v in vars(args).items():
    print(' ' * 26 + k + ': ' + str(v))


# Setup
results_dir = os.path.join('results', '{}_{}'.format(args.env, args.id))
os.makedirs(results_dir, exist_ok=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() and not args.disable_cuda:
    print("using CUDA")
    args.device = torch.device('cuda')
    torch.cuda.manual_seed(args.seed)
else:
    print("using CPU")
    args.device = torch.device('cpu')
# metrics['episodes']：记录了每个 episode 的编号。例如，如果已经完成了 10 个 episode，那么 metrics['episodes'] 的值可能是 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# metrics['steps']：记录了每个 episode 结束时的累计步数。例如，如果每个 episode 的步数分别是 100, 200, 150, ...，那么 metrics['steps'] 的值可能是 [100, 300, 450, ...]
metrics = {
    'steps': [],
    'episodes': [],
    'train_rewards': [],
    'test_episodes': [],
    'test_rewards': [],
    'observation_loss': [],
    'reward_loss': [],
    'kl_loss': [],
    'actor_loss': [],
    'value_loss': [],
}

summary_name = results_dir + "/{}_{}_log"
writer = SummaryWriter(summary_name.format(args.env, args.id))
print("writer is ready")

# Initialise training environment and experience replay memory
# todo 先只看gym环境
env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth)
print("environment is loaded")
if args.experience_replay != '' and os.path.exists(args.experience_replay):
    # 非必需
    '''
    经验回放（Experience Replay）并不是 Dreamer 算法的核心要求，因为 Dreamer 主要依靠构建环境的世界模型，再利用模型进行想象（imagination）来生成训练数据。然而，这里加上 Experience Replay 的原因包括：

    1. **数据高效性**  
    经验回放允许重复利用已经采集的真实交互数据，从而提高样本利用率，对训练更稳定有帮助。

    2. **降低数据相关性**  
    将收集到的经验存入缓冲区，再随机采样批次训练能够打破时间相关性，使训练数据更独立。

    3. **稳定性与收敛性**  
    在实际应用中，即使是模型基的 RL 算法，通过 Replay Buffer 可以平滑环境噪声，提高训练的稳定性。

    总结来说，虽然 Dreamer 本身可以利用从模型想象所得的数据进行训练，但现实中引入经验回放可以更充分利用真实数据，加速和稳定模型学习。
    '''
    D = torch.load(args.experience_replay)
    metrics['steps'], metrics['episodes'] = [D.steps] * D.episodes, list(range(1, D.episodes + 1))
elif not args.test:
    # 构建经验回放缓冲区
    D = ExperienceReplay(
        args.experience_size, args.symbolic_env, env.observation_size, env.action_size, args.bit_depth, args.device
    )
    # Initialise dataset D with S random seed episodes
    # 初始化预热经验重放缓存中，使用随机策略采集 seed_episodes 个序列数据
    # 并记录每个序列的部署
    for s in range(1, args.seed_episodes + 1):
        observation, done, t = env.reset(), False, 0
        while not done:
            action = env.sample_random_action()
            next_observation, reward, done = env.step(action)
            D.append(observation, action, reward, done)
            observation = next_observation
            t += 1
        metrics['steps'].append(t * args.action_repeat + (0 if len(metrics['steps']) == 0 else metrics['steps'][-1]))
        metrics['episodes'].append(s)
print("experience replay buffer is ready")


# Initialise model parameters randomly
# todo 什么事belief_size
transition_model = TransitionModel(
    args.belief_size,
    args.state_size,
    env.action_size,
    args.hidden_size,
    args.embedding_size,
    args.dense_activation_function,
).to(device=args.device)
# 观察模型很简单，只是简单的一个全连接层，用于将信念和隐状态映射到观察值
# 但是观察具备前后相关性，所以就需要一个TransitionModel来处理 todo 是吗？
observation_model = ObservationModel(
    args.symbolic_env,
    env.observation_size,
    args.belief_size,
    args.state_size,
    args.embedding_size,
    args.cnn_activation_function,
).to(device=args.device)
reward_model = RewardModel(args.belief_size, args.state_size, args.hidden_size, args.dense_activation_function).to(
    device=args.device
)
encoder = Encoder(args.symbolic_env, env.observation_size, args.embedding_size, args.cnn_activation_function).to(
    device=args.device
)
actor_model = ActorModel(
    args.belief_size, args.state_size, args.hidden_size, env.action_size, args.dense_activation_function
).to(device=args.device)
value_model = ValueModel(args.belief_size, args.state_size, args.hidden_size, args.dense_activation_function).to(
    device=args.device
)
param_list = (
    list(transition_model.parameters())
    + list(observation_model.parameters())
    + list(reward_model.parameters())
    + list(encoder.parameters())
)
value_actor_param_list = list(value_model.parameters()) + list(actor_model.parameters())
params_list = param_list + value_actor_param_list
print("transition, observation, reward, encoder, actor, value models are ready")
model_optimizer = optim.Adam(
    param_list, lr=0 if args.learning_rate_schedule != 0 else args.model_learning_rate, eps=args.adam_epsilon
)
actor_optimizer = optim.Adam(
    actor_model.parameters(),
    lr=0 if args.learning_rate_schedule != 0 else args.actor_learning_rate,
    eps=args.adam_epsilon,
)
value_optimizer = optim.Adam(
    value_model.parameters(),
    lr=0 if args.learning_rate_schedule != 0 else args.value_learning_rate,
    eps=args.adam_epsilon,
)
if args.models != '' and os.path.exists(args.models):
    print("loading pre-trained models")
    model_dicts = torch.load(args.models)
    transition_model.load_state_dict(model_dicts['transition_model'])
    observation_model.load_state_dict(model_dicts['observation_model'])
    reward_model.load_state_dict(model_dicts['reward_model'])
    encoder.load_state_dict(model_dicts['encoder'])
    actor_model.load_state_dict(model_dicts['actor_model'])
    value_model.load_state_dict(model_dicts['value_model'])
    model_optimizer.load_state_dict(model_dicts['model_optimizer'])
if args.algo == "dreamer":
    print("DREAMER")
    planner = actor_model
else:
    '''
    todo 这段先不看
    '''
    print("PLANET")
    planner = MPCPlanner(
        env.action_size,
        args.planning_horizon,
        args.optimisation_iters,
        args.candidates,
        args.top_candidates,
        transition_model,
        reward_model,
    )

# 一个全局的先验分布，用于计算 KL 散度。它通常被设定为标准正态分布（均值为 0，方差为 1）。在 Dreamer 算法中，global_prior 用于计算全局 KL 散度损失，从而帮助模型在训练过程中保持稳定。
global_prior = Normal(
    torch.zeros(args.batch_size, args.state_size, device=args.device),
    torch.ones(args.batch_size, args.state_size, device=args.device),
)  # Global prior N(0, I)

#  是一个用于限制 KL 散度的参数。它的作用是防止 KL 散度过大，从而避免模型过度拟合。具体来说，free_nats 允许 KL 散度在一定范围内自由变化，而不会对模型的损失函数产生影响
free_nats = torch.full((1,), args.free_nats, device=args.device)  # Allowed deviation in KL divergence
print("models and planners are ready")


'''
函数在 Dreamer 算法中用于更新智能体的信念状态和隐状态，并选择和执行动作。该函数结合了当前的观察值、动作和模型，推断出当前的信念状态和隐状态，并根据策略选择新的动作，然后在环境中执行该动作，获取新的观察值和奖励
'''
def update_belief_and_act(
    args, env, planner, transition_model, encoder, belief, posterior_state, action, observation, explore=False
):
    '''
    args：包含各种超参数的命令行参数。
    env：环境对象。
    planner：策略模型（在 Dreamer 中通常是 Actor 模型）。
    transition_model：状态转移模型。
    encoder：编码器模型，用于将观察值编码为特征向量。
    belief：当前的信念状态。
    posterior_state：当前的隐状态。
    action：当前的动作。
    observation：当前的观察值。
    explore：布尔值，指示是否进行探索（添加噪声）
    '''
    # Infer belief over current state q(s_t|o≤t,a<t) from the history
    # print("action size: ",action.size()) torch.Size([1, 6])
    # 使用状态转移模型（Transition Model）结合当前的隐状态、动作和编码后的观察值，推断出新的信念状态和隐状态
    # 这里的 encoder(observation).unsqueeze(dim=0) 将观察值编码为特征向量，并添加时间维度（因为只有一个t，所以第一个维度unsqueeze(dim=0)变成1）
    belief, _, _, _, posterior_state, _, _ = transition_model(
        posterior_state, action.unsqueeze(dim=0), belief, encoder(observation).unsqueeze(dim=0)
    )  # Action and observation need extra time dimension
    # 移除信念状态和隐状态中的时间维度。
    belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(
        dim=0
    )  # Remove time dimension from belief/state
    if args.algo == "dreamer":
        # todo 这里感觉是错误 ，重新采集数据是使用的是explore为True，而测试时使用的是False
        action = planner.get_action(belief, posterior_state, det=not (explore))
    else:
        action = planner(belief, posterior_state)  # Get action from planner(q(s_t|o≤t,a<t), p)
    if explore:
        # 如果为True则给动作进行采样
        action = torch.clamp(
            Normal(action, args.action_noise).rsample(), -1, 1
        )  # Add gaussian exploration noise on top of the sampled action
        # action = action + args.action_noise * torch.randn_like(action)  # Add exploration noise ε ~ p(ε) to the action
    # 然后执行动作得到下一个
    next_observation, reward, done = env.step(
        action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu()
    )  # Perform environment step (action repeats handled internally)
    # 返回更新后的信念状态、隐状态、动作、下一个观察值、奖励和是否结束标志
    return belief, posterior_state, action, next_observation, reward, done


# Testing only
if args.test:
    # Set models to eval mode
    transition_model.eval()
    reward_model.eval()
    encoder.eval()
    with torch.no_grad():
        total_reward = 0
        for _ in tqdm(range(args.test_episodes)):
            observation = env.reset()
            belief, posterior_state, action = (
                torch.zeros(1, args.belief_size, device=args.device),
                torch.zeros(1, args.state_size, device=args.device),
                torch.zeros(1, env.action_size, device=args.device),
            )
            pbar = tqdm(range(args.max_episode_length // args.action_repeat))
            for t in pbar:
                belief, posterior_state, action, observation, reward, done = update_belief_and_act(
                    args,
                    env,
                    planner,
                    transition_model,
                    encoder,
                    belief,
                    posterior_state,
                    action,
                    observation.to(device=args.device),
                )
                total_reward += reward
                if args.render:
                    env.render()
                if done:
                    pbar.close()
                    break
    print('Average Reward:', total_reward / args.test_episodes)
    env.close()
    quit()


# Training (and testing)
# 这里是为了可持续化训练设置的一个循环，每次循环都会执行以下操作，每次训练都从上一次的训练结束的地方开始
# todo 每一个episode都是游戏过程吗？
for episode in tqdm(
    range(metrics['episodes'][-1] + 1, args.episodes + 1), total=args.episodes, initial=metrics['episodes'][-1] + 1
):
    # Model fitting
    losses = []
    model_modules = transition_model.modules + encoder.modules + observation_model.modules + reward_model.modules

    print("training loop")
    # todo collect_interval的作用，是用来控制训练的次数的吗？
    # 这边看起来应该是控制训练的次数
    for s in tqdm(range(args.collect_interval)):
        # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
        # 采集环境数据，格式：【【a0, b0, c0...Nchunk_size】，【a1, b1, c1】，【a2, b2, c2】，【a3, b3, c3】... [Abatch_size, Bbatch_size...]】
        # shape (time, batch, features)
        observations, actions, rewards, nonterminals = D.sample(
            args.batch_size, args.chunk_size
        )  # Transitions start at time t = 0
        # Create initial belief and state for time t = 0
        # todo 总结什么事信念状态，这里的状态是什么状态
        # 初始的隐状态（latent state），它也是一个高维的向量，用于表示环境的潜在状态。隐状态通过模型的状态转移网络（Transition Model）来更新，并结合当前的信念状态和动作，预测下一个隐状态。
        init_belief, init_state = torch.zeros(args.batch_size, args.belief_size, device=args.device), torch.zeros(
            args.batch_size, args.state_size, device=args.device
        )
        # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
        # actions[:-1] 不传入最后一个动作
        # observations[1:] 不传入第一个观察值
        # nonterminals[:-1] 不传入最后一个是否结束
        # bottle(encoder, (observations[1:],)): 将环境观察提取特征，shape变成（time, batch, embed_features）
        (
            beliefs,
            prior_states,
            prior_means,
            prior_std_devs,
            posterior_states,
            posterior_means,
            posterior_std_devs,
        ) = transition_model(
            init_state, actions[:-1], init_belief, bottle(encoder, (observations[1:],)), nonterminals[:-1]
        )
        # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
        # 计算观察值损失
        # 对数概率损失：

        # 对数概率损失可以更好地处理观测值的概率分布，特别是在观测值具有不确定性或噪声的情况下。
        # 通过计算观测值在模型生成的概率分布下的对数概率，可以更准确地衡量模型生成的观测值与实际观测值之间的匹配程度。
        # 均方误差损失：

        # 均方误差损失是一种常见的回归损失函数，用于衡量模型生成的观测值与实际观测值之间的差异。
        # 这种损失函数简单且易于计算，适用于观测值具有较小噪声或不确定性的情况

        # 
        if args.worldmodel_LogProbLoss:
            # 对数概率损失，这里使用观察模型根据信念和后验状态生成预测的观察值的均值，然后构建一个正太分布
            observation_dist = Normal(bottle(observation_model, (beliefs, posterior_states)), 1)
            # 较大的对数概率表示模型生成的观察值与实际观察值非常匹配，而较小的对数概率表示不匹配
            # todo log_prob这部分实在做什么数学计算？数学公式有哪些？
            observation_loss = (
                -observation_dist.log_prob(observations[1:])
                .sum(dim=2 if args.symbolic_env else (2, 3, 4))
                .mean(dim=(0, 1))
            )
        else:
            # 均方误差损失
            observation_loss = (
                F.mse_loss(bottle(observation_model, (beliefs, posterior_states)), observations[1:], reduction='none')
                .sum(dim=2 if args.symbolic_env else (2, 3, 4))
                .mean(dim=(0, 1))  
            )

        # 这边也是开始计算奖励损失，原理和观察值损失一样
        if args.worldmodel_LogProbLoss:
            reward_dist = Normal(bottle(reward_model, (beliefs, posterior_states)), 1)
            reward_loss = -reward_dist.log_prob(rewards[:-1]).mean(dim=(0, 1))
        else:
            reward_loss = F.mse_loss(
                bottle(reward_model, (beliefs, posterior_states)), rewards[:-1], reduction='none'
            ).mean(dim=(0, 1))
        # transition loss
        # todo 这个散度计算的作用？
        '''
        KL 散度的作用：

        正则化：KL 散度作为一种正则化项，限制了后验分布与先验分布之间的差异，防止模型过度拟合训练数据。
        稳定性：通过限制隐状态的分布，KL 散度有助于提高模型训练的稳定性，使得模型在面对新数据时能够更好地泛化。
        信息约束：KL 散度确保模型在隐状态中编码的信息量适中，不会过多或过少，从而提高模型的表达能力。
        为什么要使用先验和后验计算 KL 散度：

        先验分布：先验分布通常是一个简单的分布（如标准正态分布），它表示在没有观察到数据时对隐状态的先验假设。
        后验分布：后验分布结合了观察到的数据，表示在给定数据的情况下对隐状态的更新后的估计。
        KL 散度：通过计算后验分布与先验分布之间的 KL 散度，可以衡量模型在观察到数据后对隐状态的更新程度。较小的 KL 散度表示模型生成的隐状态分布与先验分布较为一致，从而避免过拟合。

        后验分布结合了观测数据，而先验分布通常是一个简单的参考分布（如标准正态）。通过 KL 散度约束两者之间的差异，可以避免后验分布过度拟合观测数据、偏离先验假设，从而在训练中保持模型分布的稳定性，并帮助模型更好地泛化到新数据。
        '''
        div = kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2)
        # 限制散度的大小torch.max(div, free_nats)
        # 这个应该是在训练transition_model模型
        kl_loss = torch.max(div, free_nats).mean(
            dim=(0, 1)
        )  # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out
        if args.global_kl_beta != 0:
            # 如果设置了全局 KL 散度权重（global_kl_beta），则计算后验分布与全局先验分布（global_prior）之间的 KL 散度，并将其加到 kl_loss 中
            kl_loss += args.global_kl_beta * kl_divergence(
                Normal(posterior_means, posterior_std_devs), global_prior
            ).sum(dim=2).mean(dim=(0, 1))
        # Calculate latent overshooting objective for t > 0
        if args.overshooting_kl_beta != 0:
            # 计算超前 KL 散度损失 todo 啥事超前
            overshooting_vars = []  # Collect variables for overshooting to process in batch
            # 采集超前数据
            for t in range(1, args.chunk_size - 1):
                # 限制d不会超过范围
                # overshooting_distance应该是一个超前的距离大小
                d = min(t + args.overshooting_distance, args.chunk_size - 1)  # Overshooting distance
                # 前一个时间步
                t_, d_ = t - 1, d - 1  # Use t_ and d_ to deal with different time indexing for latent states
                seq_pad = (
                    0,
                    0,
                    0,
                    0,
                    0,
                    t - d + args.overshooting_distance,
                )  # Calculate sequence padding so overshooting terms can be calculated in one batch
                # Store (0) actions, (1) nonterminals, (2) rewards, (3) beliefs, (4) prior states, (5) posterior means, (6) posterior standard deviations and (7) sequence masks
                # 最后一个起到掩码的作用 防止考虑到不存在的数值
                overshooting_vars.append(
                    (
                        F.pad(actions[t:d], seq_pad),
                        F.pad(nonterminals[t:d], seq_pad),
                        F.pad(rewards[t:d], seq_pad[2:]),
                        beliefs[t_],
                        prior_states[t_],
                        F.pad(posterior_means[t_ + 1 : d_ + 1].detach(), seq_pad),
                        F.pad(posterior_std_devs[t_ + 1 : d_ + 1].detach(), seq_pad, value=1),
                        F.pad(torch.ones(d - t, args.batch_size, args.state_size, device=args.device), seq_pad),
                    )
                )  # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences
            overshooting_vars = tuple(zip(*overshooting_vars))
            # Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
            beliefs, prior_states, prior_means, prior_std_devs = transition_model(
                torch.cat(overshooting_vars[4], dim=0), # prior_states
                torch.cat(overshooting_vars[0], dim=1), # actions
                torch.cat(overshooting_vars[3], dim=0), # beliefs
                None, # 不考虑后验状态
                torch.cat(overshooting_vars[1], dim=1), # nonterminals
            )
            seq_mask = torch.cat(overshooting_vars[7], dim=1)
            # Calculate overshooting KL loss with sequence mask
            # 超前kl散度是考虑了超前的状态
            # 计算后验分布与先验分布之间的 KL 散度，并使用序列掩码进行加权。
            # 使用 torch.max 函数确保 KL 散度不会小于 free_nats，从而避免 KL 散度对模型训练产生过大的影响
            kl_loss += (
                (1 / args.overshooting_distance)
                * args.overshooting_kl_beta
                * torch.max(
                    (
                        kl_divergence(
                            Normal(torch.cat(overshooting_vars[5], dim=1), torch.cat(overshooting_vars[6], dim=1)),
                            Normal(prior_means, prior_std_devs),
                        )
                        * seq_mask
                    ).sum(dim=2),
                    free_nats,
                ).mean(dim=(0, 1))
                * (args.chunk_size - 1)
            )  # Update KL loss (compensating for extra average over each overshooting/open loop sequence)
            # Calculate overshooting reward prediction loss with sequence mask
            # 计算模型生成的奖励与实际奖励之间的均方误差，并使用序列掩码进行加权
            if args.overshooting_reward_scale != 0:
                reward_loss += (
                    (1 / args.overshooting_distance)
                    * args.overshooting_reward_scale
                    * F.mse_loss(
                        bottle(reward_model, (beliefs, prior_states)) * seq_mask[:, :, 0],
                        torch.cat(overshooting_vars[2], dim=1),
                        reduction='none',
                    ).mean(dim=(0, 1))
                    * (args.chunk_size - 1)
                )  # Update reward loss (compensating for extra average over each overshooting/open loop sequence)
        # Apply linearly ramping learning rate schedule
        # 学习率调度
        if args.learning_rate_schedule != 0:
            for group in model_optimizer.param_groups:
                group['lr'] = min(
                    group['lr'] + args.model_learning_rate / args.model_learning_rate_schedule, args.model_learning_rate
                )
        # 训练模型
        model_loss = observation_loss + reward_loss + kl_loss
        # Update model parameters
        model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(param_list, args.grad_clip_norm, norm_type=2)
        model_optimizer.step()

        # 上面转移、观察、奖励等环境模型训练完成后，开始训练动作、评价模型
        # Dreamer implementation: actor loss calculation and optimization
        with torch.no_grad():
            actor_states = posterior_states.detach()
            actor_beliefs = beliefs.detach()
        with FreezeParameters(model_modules):
            # 选中的部分代码是在进行想象轨迹（imagination trajectory）的生成。具体来说，它使用冻结的模型参数，通过在隐空间中进行前向传播，生成未来的信念状态和隐状态。这些想象轨迹用于训练 Actor 模型和 Value 模型
            imagination_traj = imagine_ahead(
                actor_states, actor_beliefs, actor_model, transition_model, args.planning_horizon
            )
        # 根据提取特征后的状态和动作信念、动作网络预测得到想象的信念、先验状态、均值和标准差
        imged_beliefs, imged_prior_states, imged_prior_means, imged_prior_std_devs = imagination_traj
        with FreezeParameters(model_modules + value_model.modules):
            # 根据想象的信念、先验状态、均值和标准差，以及动作网络预测的奖励，计算奖励预测和值函数预测
            imged_reward = bottle(reward_model, (imged_beliefs, imged_prior_states))
            value_pred = bottle(value_model, (imged_beliefs, imged_prior_states))
        # 得到每个时间步的回报 return
        # todo 这里传入的需要是连续轨迹，那么记录一下实际的shape什么样子的，我记得采样时随机块位置的吧
        returns = lambda_return(
            imged_reward, value_pred, bootstrap=value_pred[-1], discount=args.discount, lambda_=args.disclam
        )

        # 计算 actor 损失，这里使用了负的回报作为损失，目标是最大化回报，所以用来优化动作
        actor_loss = -torch.mean(returns)
        # Update model parameters
        actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor_model.parameters(), args.grad_clip_norm, norm_type=2)
        actor_optimizer.step()

        # Dreamer implementation: value loss calculation and optimization
        with torch.no_grad():
            value_beliefs = imged_beliefs.detach()
            value_prior_states = imged_prior_states.detach()
            target_return = returns.detach()

        # 根据评价模型得到一个评价正太分布，然后计算损失
        # 训练评价模型
        value_dist = Normal(
            bottle(value_model, (value_beliefs, value_prior_states)), 1
        )  # detach the input tensor from the transition network.
        # 评价不能偏离目标回报太远
        value_loss = -value_dist.log_prob(target_return).mean(dim=(0, 1))
        # Update model parameters
        value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(value_model.parameters(), args.grad_clip_norm, norm_type=2)
        value_optimizer.step()

        # # Store (0) observation loss (1) reward loss (2) KL loss (3) actor loss (4) value loss
        losses.append(
            [observation_loss.item(), reward_loss.item(), kl_loss.item(), actor_loss.item(), value_loss.item()]
        )
    
    # 完成每次训练后开始更新数据
    # Update and plot loss metrics
    losses = tuple(zip(*losses))
    metrics['observation_loss'].append(losses[0])
    metrics['reward_loss'].append(losses[1])
    metrics['kl_loss'].append(losses[2])
    metrics['actor_loss'].append(losses[3])
    metrics['value_loss'].append(losses[4])
    lineplot(
        metrics['episodes'][-len(metrics['observation_loss']) :],
        metrics['observation_loss'],
        'observation_loss',
        results_dir,
    )
    lineplot(metrics['episodes'][-len(metrics['reward_loss']) :], metrics['reward_loss'], 'reward_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['kl_loss']) :], metrics['kl_loss'], 'kl_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['actor_loss']) :], metrics['actor_loss'], 'actor_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['value_loss']) :], metrics['value_loss'], 'value_loss', results_dir)

    # Data collection
    # 这里重新开始采集环境数据，采集一次，训练100次，提高数据的利用率
    print("Data collection")
    with torch.no_grad():
        # 初始化环境
        observation, total_reward = env.reset(), 0
        # 初始化信念和后验状态以及动作
        belief, posterior_state, action = (
            torch.zeros(1, args.belief_size, device=args.device),
            torch.zeros(1, args.state_size, device=args.device),
            torch.zeros(1, env.action_size, device=args.device),
        )
        # action_repeat 动作重复次数
        # max_episode_length：每次episode的最大长度
        pbar = tqdm(range(args.max_episode_length // args.action_repeat))
        # 得到要执行几次东走
        for t in pbar:
            # print("step",t)
            # planner：动作规划器策略
            # encoder：观察提取特征
            # transition_model：转移模型

            # 这里actoin和observation对应关系没错，observation表示当前的观察，action表示上一个观察的动作（初始时上一个观察的动作就是0）
            # 在 Dreamer 中，**后验状态**（posterior_state）才是融合了真实观测信息后的隐状态估计，因此在下一步中继续使用它能让模型拥有更准确的环境感知。**先验状态**（prior_state）仅基于上一时刻预测，不包含当前观测，准确度较低。由于算法每一步都要结合真实观测来更新状态，因此会将后验状态再次作为输入，以便保持对当前环境的最佳估计。
            # todo 为什么这里和训练时不同
            belief, posterior_state, action, next_observation, reward, done = update_belief_and_act(
                args,
                env,
                planner,
                transition_model,
                encoder,
                belief,
                posterior_state,
                action,
                observation.to(device=args.device),
                explore=True,
            )
            D.append(observation, action.cpu(), reward, done)
            total_reward += reward
            observation = next_observation
            if args.render:
                env.render()
            if done:
                pbar.close()
                break

        # Update and plot train reward metrics
        metrics['steps'].append(t + metrics['steps'][-1])
        metrics['episodes'].append(episode)
        metrics['train_rewards'].append(total_reward)
        lineplot(
            metrics['episodes'][-len(metrics['train_rewards']) :],
            metrics['train_rewards'],
            'train_rewards',
            results_dir,
        )

    # Test model
    print("Test model")
    if episode % args.test_interval == 0:
        # Set models to eval mode
        transition_model.eval()
        observation_model.eval()
        reward_model.eval()
        encoder.eval()
        actor_model.eval()
        value_model.eval()
        # Initialise parallelised test environments
        test_envs = EnvBatcher(
            Env,
            (args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth),
            {},
            args.test_episodes,
        )

        with torch.no_grad():
            observation, total_rewards, video_frames = test_envs.reset(), np.zeros((args.test_episodes,)), []
            belief, posterior_state, action = (
                torch.zeros(args.test_episodes, args.belief_size, device=args.device),
                torch.zeros(args.test_episodes, args.state_size, device=args.device),
                torch.zeros(args.test_episodes, env.action_size, device=args.device),
            )
            pbar = tqdm(range(args.max_episode_length // args.action_repeat))
            for t in pbar:
                belief, posterior_state, action, next_observation, reward, done = update_belief_and_act(
                    args,
                    test_envs,
                    planner,
                    transition_model,
                    encoder,
                    belief,
                    posterior_state,
                    action,
                    observation.to(device=args.device),
                )
                total_rewards += reward.numpy()
                if not args.symbolic_env:  # Collect real vs. predicted frames for video
                    video_frames.append(
                        make_grid(
                            torch.cat([observation, observation_model(belief, posterior_state).cpu()], dim=3) + 0.5,
                            nrow=5,
                        ).numpy()
                    )  # Decentre
                observation = next_observation
                if done.sum().item() == args.test_episodes:
                    pbar.close()
                    break

        # Update and plot reward metrics (and write video if applicable) and save metrics
        metrics['test_episodes'].append(episode)
        metrics['test_rewards'].append(total_rewards.tolist())
        lineplot(metrics['test_episodes'], metrics['test_rewards'], 'test_rewards', results_dir)
        lineplot(
            np.asarray(metrics['steps'])[np.asarray(metrics['test_episodes']) - 1],
            metrics['test_rewards'],
            'test_rewards_steps',
            results_dir,
            xaxis='step',
        )
        if not args.symbolic_env:
            episode_str = str(episode).zfill(len(str(args.episodes)))
            write_video(video_frames, 'test_episode_%s' % episode_str, results_dir)  # Lossy compression
            save_image(
                torch.as_tensor(video_frames[-1]), os.path.join(results_dir, 'test_episode_%s.png' % episode_str)
            )
        torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

        # Set models to train mode
        transition_model.train()
        observation_model.train()
        reward_model.train()
        encoder.train()
        actor_model.train()
        value_model.train()
        # Close test environments
        test_envs.close()

    writer.add_scalar("train_reward", metrics['train_rewards'][-1], metrics['steps'][-1])
    writer.add_scalar("train/episode_reward", metrics['train_rewards'][-1], metrics['steps'][-1] * args.action_repeat)
    writer.add_scalar("observation_loss", metrics['observation_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar("reward_loss", metrics['reward_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar("kl_loss", metrics['kl_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar("actor_loss", metrics['actor_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar("value_loss", metrics['value_loss'][0][-1], metrics['steps'][-1])
    print(
        "episodes: {}, total_steps: {}, train_reward: {} ".format(
            metrics['episodes'][-1], metrics['steps'][-1], metrics['train_rewards'][-1]
        )
    )

    # Checkpoint models
    if episode % args.checkpoint_interval == 0:
        torch.save(
            {
                'transition_model': transition_model.state_dict(),
                'observation_model': observation_model.state_dict(),
                'reward_model': reward_model.state_dict(),
                'encoder': encoder.state_dict(),
                'actor_model': actor_model.state_dict(),
                'value_model': value_model.state_dict(),
                'model_optimizer': model_optimizer.state_dict(),
                'actor_optimizer': actor_optimizer.state_dict(),
                'value_optimizer': value_optimizer.state_dict(),
            },
            os.path.join(results_dir, 'models_%d.pth' % episode),
        )
        if args.checkpoint_experience:
            torch.save(
                D, os.path.join(results_dir, 'experience.pth')
            )  # Warning: will fail with MemoryError with large memory sizes


# Close training environment
env.close()
