from typing import List, Optional

import numpy as np
import torch
import torch.distributions
from torch import jit, nn
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.nn import functional as F


# Wraps the input tuple for a function to process a time x batch x features sequence in batch x features (assumes one output)
def bottle(f, x_tuple):
    '''
    方法的作用是将输入的时间序列数据从 (time, batch, features) 的形状转换为 (batch * time, features) 的形状，以便于在神经网络中进行批处理操作，然后再将输出转换回原来的形状 (time, batch, features)。
    '''
    # map(lambda x: x.size(), x_tuple) 为获取x_tuple中每个元素的size=(time, batch, features): x_sizes
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    # x[1][0]=time, x[1][1]=batch, x[1][2]=features
    # (time, batch, features) 的形状转换为 (batch * time, features)
    # 再传入encoder中
    # shape of x_tuple: (time, batch, embeddin_features)
    y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
    y_size = y.size()
    # (batch * time, features) 的形状转换为 (time, batch, embeddin_features)
    output = y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])
    return output


class TransitionModel(jit.ScriptModule):
    '''
    Dreamer 算法中的一个关键组件，它用于预测环境状态的转移。具体来说，TransitionModel 的作用包括：

    状态转移预测：

    根据当前的隐状态（latent state）和动作，预测下一个隐状态。这是通过一个递归神经网络（如 LSTM 或 GRU）来实现的。
    隐状态的先验和后验分布：

    计算隐状态的先验分布（prior distribution）和后验分布（posterior distribution）。先验分布是基于当前隐状态和动作预测的，而后验分布则结合了实际观察到的环境状态。
    生成模型：

    作为生成模型的一部分，TransitionModel 与 ObservationModel 和 RewardModel 一起工作，用于生成未来的观察值和奖励。这对于模型的训练和规划（planning）非常重要。
    '''
    __constants__ = ['min_std_dev']

    def __init__(
        self,
        belief_size,
        state_size,
        action_size,
        hidden_size,
        embedding_size,
        activation_function='relu',
        min_std_dev=0.1,
    ):
        '''
        belief_size：信念状态的大小。
        state_size：隐状态的大小。
        action_size：动作的大小。
        hidden_size：隐藏层的大小。
        embedding_size：嵌入层的大小。
        dense_activation_function：密集层的激活函数。
        '''
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        self.fc_embed_state_action = nn.Linear(state_size + action_size, belief_size)
        self.rnn = nn.GRUCell(belief_size, belief_size)
        self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
        self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)
        self.fc_embed_belief_posterior = nn.Linear(belief_size + embedding_size, hidden_size)
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)
        self.modules = [
            self.fc_embed_state_action,
            self.fc_embed_belief_prior,
            self.fc_state_prior,
            self.fc_embed_belief_posterior,
            self.fc_state_posterior,
        ]

    # Operates over (previous) state, (previous) actions, (previous) belief, (previous) nonterminals (mask), and (current) observations
    # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
    # t :  0  1  2  3  4  5
    # o :    -X--X--X--X--X-
    # a : -X--X--X--X--X-
    # n : -X--X--X--X--X-
    # pb: -X-
    # ps: -X-
    # b : -x--X--X--X--X--X-
    # s : -x--X--X--X--X--X-
    @jit.script_method
    def forward(
        self,
        prev_state: torch.Tensor,
        actions: torch.Tensor,
        prev_belief: torch.Tensor,
        observations: Optional[torch.Tensor] = None,
        nonterminals: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        '''
        prev_state: 上一个隐状态
        actions: 动作，没有传入最后一个动作，shape=(time - 1, batch, action_size)
        prev_belief: 上一个信念状态
        observations: 观察值编码后的特征状态值，没有传入第一个观察值，shape=(1:time， batch, observation_size)
        nonterminals: 非终止状态没有传入最后一个中止符号，shape=(time - 1, batch, 1)
        Input: init_belief, init_state:  torch.Size([50, 200]) torch.Size([50, 30])
        Output: beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs
                torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
        '''
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        # 得到时间步长度
        T = actions.size(0) + 1
        # 初始化隐状态prior为先验，posterior为后验
        # 初始化隐藏状态列表，用于存储每个时间步的信念状态、先验状态、先验均值、先验标准差、后验状态、后验均值和后验标准差
        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = (
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
        )
        beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state
        # Loop over time sequence
        for t in range(T - 1):
            # todo 为啥要这么选择？
            _state = (
                prior_states[t] if observations is None else posterior_states[t]
            )  # Select appropriate previous state
            # 根据中止是否有值，选择是否mask
            _state = (
                _state if nonterminals is None else _state * nonterminals[t]
            )  # Mask if previous transition was terminal
            # Compute belief (deterministic hidden state)
            # 根据状态和动作提取特征 
            hidden = self.act_fn(self.fc_embed_state_action(torch.cat([_state, actions[t]], dim=1)))
            #然后通过GRUCell进行更新 到后一个信念状态
            beliefs[t + 1] = self.rnn(hidden, beliefs[t])
            # Compute state prior by applying transition dynamics
            # 通过信念状态提取特征
            hidden = self.act_fn(self.fc_embed_belief_prior(beliefs[t + 1]))
            # 计算先验均值和标准差
            prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
            prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
            # 根据先验均值和标准差采样得到下一个先验状态，使用随机噪声采样
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])
            if observations is not None:
                # Compute state posterior by applying transition dynamics and using current observation
                t_ = t - 1  # Use t_ to deal with different time indexing for observations
                # 根据上一个观察值（因为传入的观察和动作差一个时间单位），和后一个信念状态提取特征
                hidden = self.act_fn(
                    self.fc_embed_belief_posterior(torch.cat([beliefs[t + 1], observations[t_ + 1]], dim=1))
                )
                # 根据提取的观察特征计算后验均值和标准差
                posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
                posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
                posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(
                    posterior_means[t + 1]
                )
            #通过以上不断循环，得到了每个时间步的信念状态、先验状态、先验均值、先验标准差、后验状态、后验均值和后验标准差
        # Return new hidden states
        # 根据以上可知0时刻基本都是没有的，所以返回的时候从1开始
        # 依次返回信念状态、先验状态、先验均值、先验标准差、后验状态、后验均值和后验标准差，均是和动作以及是否结束结合计算
        hidden = [
            torch.stack(beliefs[1:], dim=0),
            torch.stack(prior_states[1:], dim=0),
            torch.stack(prior_means[1:], dim=0),
            torch.stack(prior_std_devs[1:], dim=0),
        ]
        if observations is not None:
            # 如果有观察值，则返回后验状态、后验均值和后验标准差，均是和观察值结合计算
            hidden += [
                torch.stack(posterior_states[1:], dim=0),
                torch.stack(posterior_means[1:], dim=0),
                torch.stack(posterior_std_devs[1:], dim=0),
            ]
        return hidden


class SymbolicObservationModel(jit.ScriptModule):
    '''
    是 Dreamer 算法中的一个关键组件，它用于从隐状态（latent state）和信念状态（belief state）生成观察值（observations）。具体来说，ObservationModel 的作用包括：

生成观察值：

根据给定的隐状态和信念状态，生成对应的观察值。这通常通过一个神经网络来实现，该网络将隐状态和信念状态作为输入，并输出预测的观察值。
重建误差计算：

在训练过程中，ObservationModel 用于计算重建误差（reconstruction error），即模型生成的观察值与实际观察值之间的差异。这种误差用于指导模型的训练，使其能够更准确地预测环境的状态。
作为生成模型的一部分：

ObservationModel 与 TransitionModel 和 RewardModel 一起工作，构成了 Dreamer 算法的生成模型。生成模型用于在隐空间中进行模拟和规划，从而指导智能体的行为。
    '''
    def __init__(self, observation_size, belief_size, state_size, embedding_size, activation_function='relu'):
        '''
        observation_size：观察值的大小。
        belief_size：信念状态的大小。
        state_size：隐状态的大小。
        embedding_size：嵌入层的大小。
        cnn_activation_function：卷积层的激活函数。
        '''
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, observation_size)
        self.modules = [self.fc1, self.fc2, self.fc3]

    @jit.script_method
    def forward(self, belief, state):
        hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        observation = self.fc3(hidden)
        return observation


class VisualObservationModel(jit.ScriptModule):
    __constants__ = ['embedding_size']

    def __init__(self, belief_size, state_size, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)
        self.modules = [self.fc1, self.conv1, self.conv2, self.conv3, self.conv4]

    @jit.script_method
    def forward(self, belief, state):
        hidden = self.fc1(torch.cat([belief, state], dim=1))  # No nonlinearity here
        hidden = hidden.view(-1, self.embedding_size, 1, 1)
        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        observation = self.conv4(hidden)
        return observation


def ObservationModel(symbolic, observation_size, belief_size, state_size, embedding_size, activation_function='relu'):
    if symbolic:
        return SymbolicObservationModel(observation_size, belief_size, state_size, embedding_size, activation_function)
    else:
        return VisualObservationModel(belief_size, state_size, embedding_size, activation_function)


class RewardModel(jit.ScriptModule):
    '''
     Dreamer 算法中的一个关键组件，它用于从隐状态（latent state）和信念状态（belief state）生成奖励值（rewards）。具体来说，RewardModel 的作用包括：

    生成奖励值：

    根据给定的隐状态和信念状态，生成对应的奖励值。这通常通过一个神经网络来实现，该网络将隐状态和信念状态作为输入，并输出预测的奖励值。
    奖励预测误差计算：

    在训练过程中，RewardModel 用于计算奖励预测误差（reward prediction error），即模型生成的奖励值与实际奖励值之间的差异。这种误差用于指导模型的训练，使其能够更准确地预测环境的奖励。
    作为生成模型的一部分：

    RewardModel 与 TransitionModel 和 ObservationModel 一起工作，构成了 Dreamer 算法的生成模型。生成模型用于在隐空间中进行模拟和规划，从而指导智能体的行为。
    '''
    def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
        '''
        belief_size：信念状态的大小。
        state_size：隐状态的大小。
        hidden_size：隐藏层的大小。
        dense_activation_function：密集层的激活函数。
        '''
        # [--belief-size: 200, --hidden-size: 200, --state-size: 30]
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.modules = [self.fc1, self.fc2, self.fc3]

    @jit.script_method
    def forward(self, belief, state):
        x = torch.cat([belief, state], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        reward = self.fc3(hidden).squeeze(dim=1)
        return reward


class ValueModel(jit.ScriptModule):
    '''
    是价值网络，用于估计给定状态的价值。它的主要作用包括：

    价值估计：

    根据当前的信念状态（belief state）和隐状态（latent state），估计对应状态的价值。这通常通过一个神经网络来实现，该网络将信念状态和隐状态作为输入，并输出状态的价值。
    价值优化：

    在训练过程中，ValueModel 通过最小化价值估计误差来优化价值函数。它使用从环境中采样的数据和模型生成的数据来更新价值参数。
    '''
    def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
        '''
        belief_size：信念状态的大小。
        state_size：隐状态的大小。
        hidden_size：隐藏层的大小。
        dense_activation_function：密集层的激活函数
        '''
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        self.modules = [self.fc1, self.fc2, self.fc3, self.fc4]

    @jit.script_method
    def forward(self, belief, state):
        x = torch.cat([belief, state], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        reward = self.fc4(hidden).squeeze(dim=1)
        return reward


class ActorModel(jit.ScriptModule):
    '''
    策略网络，用于生成智能体在给定状态下的动作。它的主要作用包括：

    动作生成：

    根据当前的信念状态（belief state）和隐状态（latent state），生成对应的动作。这通常通过一个神经网络来实现，该网络将信念状态和隐状态作为输入，并输出动作的均值和标准差。
    策略优化：

    在训练过程中，ActorModel 通过最大化预期回报来优化策略。它使用从环境中采样的数据和模型生成的数据来更新策略参数。
    '''
    def __init__(
        self,
        belief_size,
        state_size,
        hidden_size,
        action_size,
        dist='tanh_normal',
        activation_function='elu',
        min_std=1e-4,
        init_std=5,
        mean_scale=5,
    ):
        '''
        belief_size：信念状态的大小。
        state_size：隐状态的大小。
        hidden_size：隐藏层的大小。
        action_size：动作的大小。
        dense_activation_function：密集层的激活函数
        '''
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, 2 * action_size)
        self.modules = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]

        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

    @jit.script_method
    def forward(self, belief, state):
        raw_init_std = torch.log(torch.exp(self._init_std) - 1)
        x = torch.cat([belief, state], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        hidden = self.act_fn(self.fc4(hidden))
        action = self.fc5(hidden).squeeze(dim=1)

        action_mean, action_std_dev = torch.chunk(action, 2, dim=1)
        action_mean = self._mean_scale * torch.tanh(action_mean / self._mean_scale)
        action_std = F.softplus(action_std_dev + raw_init_std) + self._min_std
        return action_mean, action_std

    def get_action(self, belief, state, det=False):
        action_mean, action_std = self.forward(belief, state)
        dist = Normal(action_mean, action_std)
        dist = TransformedDistribution(dist, TanhBijector())
        dist = torch.distributions.Independent(dist, 1)
        dist = SampleDist(dist)
        if det:
            return dist.mode()
        else:
            return dist.rsample()


class SymbolicEncoder(jit.ScriptModule):
    '''
    是 Dreamer 算法中的一个组件，用于对符号表示的观察值进行编码。具体来说，SymbolicEncoder 的作用包括：

    特征提取：

    将符号表示的观察值（通常是低维的数值特征）转换为高维的嵌入表示（embedding）。这通常通过一系列全连接层（fully connected layers）来实现。
    输入预处理：

    在 Dreamer 算法中，观察值可以是符号表示的（如数值特征）或视觉表示的（如图像）。SymbolicEncoder 主要用于处理符号表示的观察值。它通过一系列全连接层（fully connected layers）将原始的符号观察值转换为高维的嵌入表示。这种嵌入表示可以更好地捕捉观察值中的重要特征，并作为后续模型（如 TransitionModel 和 ObservationModel）的输入
    '''
    def __init__(self, observation_size, embedding_size, activation_function='relu'):
        '''
        observation_size：观察值的大小。
        embedding_size：嵌入层的大小。
        activation_function：激活函数（默认为 ReLU）。
        '''
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(observation_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, embedding_size)
        self.modules = [self.fc1, self.fc2, self.fc3]

    @jit.script_method
    def forward(self, observation):
        hidden = self.act_fn(self.fc1(observation))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.fc3(hidden)
        return hidden


class VisualEncoder(jit.ScriptModule):
    __constants__ = ['embedding_size']

    def __init__(self, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)
        self.modules = [self.conv1, self.conv2, self.conv3, self.conv4]

    @jit.script_method
    def forward(self, observation):
        hidden = self.act_fn(self.conv1(observation))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = hidden.view(-1, 1024)
        hidden = self.fc(hidden)  # Identity if embedding size is 1024 else linear projection
        return hidden


def Encoder(symbolic, observation_size, embedding_size, activation_function='relu'):
    if symbolic:
        return SymbolicEncoder(observation_size, embedding_size, activation_function)
    else:
        return VisualEncoder(embedding_size, activation_function)


# "atanh", "TanhBijector" and "SampleDist" are from the following repo
# https://github.com/juliusfrost/dreamer-pytorch
def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


class TanhBijector(torch.distributions.Transform):
    def __init__(self):
        super().__init__()
        self.bijective = True
        self.domain = torch.distributions.constraints.real
        self.codomain = torch.distributions.constraints.interval(-1.0, 1.0)

    @property
    def sign(self):
        return 1.0

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y: torch.Tensor):
        y = torch.where((torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y)
        y = atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (np.log(2) - x - F.softplus(-2.0 * x))


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        sample = self._dist.rsample()
        return torch.mean(sample, 0)

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def sample(self):
        return self._dist.sample()
