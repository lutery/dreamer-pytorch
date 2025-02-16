import os
from typing import Iterable

import cv2
import numpy as np
import plotly
import torch
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
from torch.nn import Module
from torch.nn import functional as F


# Plots min, max and mean + standard deviation bars of a population over time
def lineplot(xs, ys_population, title, path='', xaxis='episode'):
    max_colour, mean_colour, std_colour, transparent = (
        'rgb(0, 132, 180)',
        'rgb(0, 172, 237)',
        'rgba(29, 202, 255, 0.2)',
        'rgba(0, 0, 0, 0)',
    )

    if isinstance(ys_population[0], list) or isinstance(ys_population[0], tuple):
        ys = np.asarray(ys_population, dtype=np.float32)
        ys_min, ys_max, ys_mean, ys_std, ys_median = ys.min(1), ys.max(1), ys.mean(1), ys.std(1), np.median(ys, 1)
        ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

        trace_max = Scatter(x=xs, y=ys_max, line=Line(color=max_colour, dash='dash'), name='Max')
        trace_upper = Scatter(x=xs, y=ys_upper, line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
        trace_mean = Scatter(
            x=xs, y=ys_mean, fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean'
        )
        trace_lower = Scatter(
            x=xs,
            y=ys_lower,
            fill='tonexty',
            fillcolor=std_colour,
            line=Line(color=transparent),
            name='-1 Std. Dev.',
            showlegend=False,
        )
        trace_min = Scatter(x=xs, y=ys_min, line=Line(color=max_colour, dash='dash'), name='Min')
        trace_median = Scatter(x=xs, y=ys_median, line=Line(color=max_colour), name='Median')
        data = [trace_upper, trace_mean, trace_lower, trace_min, trace_max, trace_median]
    else:
        data = [Scatter(x=xs, y=ys_population, line=Line(color=mean_colour))]
    plotly.offline.plot(
        {'data': data, 'layout': dict(title=title, xaxis={'title': xaxis}, yaxis={'title': title})},
        filename=os.path.join(path, title + '.html'),
        auto_open=False,
    )


def write_video(frames, title, path=''):
    frames = (
        np.multiply(np.stack(frames, axis=0).transpose(0, 2, 3, 1), 255).clip(0, 255).astype(np.uint8)[:, :, :, ::-1]
    )  # VideoWrite expects H x W x C in BGR
    _, H, W, _ = frames.shape
    writer = cv2.VideoWriter(os.path.join(path, '%s.mp4' % title), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (W, H), True)
    for frame in frames:
        writer.write(frame)
    writer.release()


def imagine_ahead(prev_state, prev_belief, policy, transition_model, planning_horizon=12):
    '''
    imagine_ahead is the function to draw the imaginary tracjectory using the dynamics model, actor, critic.
    Input: current state (posterior), current belief (hidden), policy, transition_model  # torch.Size([50, 30]) torch.Size([50, 200])
    Output: generated trajectory of features includes beliefs, prior_states, prior_means, prior_std_devs
            torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])

    imagine_ahead 是一个函数，用于使用动态模型、演员和评论家绘制想象轨迹。  
    输入：当前状态（后验），当前信念（隐藏），策略，转移模型  # torch.Size([50, 30]) torch.Size([50, 200])  
    输出：生成的特征轨迹，包括信念、先验状态、先验均值、先验标准差  ，而这里不同的就是传入的动作事动作网络预测的动作，和之前是使用真实的动作不同
    torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
    '''
    flatten = lambda x: x.view([-1] + list(x.size()[2:]))
    prev_belief = flatten(prev_belief)
    prev_state = flatten(prev_state)

    # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
    T = planning_horizon
    beliefs, prior_states, prior_means, prior_std_devs = (
        [torch.empty(0)] * T,
        [torch.empty(0)] * T,
        [torch.empty(0)] * T,
        [torch.empty(0)] * T,
    )
    beliefs[0], prior_states[0] = prev_belief, prev_state

    # Loop over time sequence
    for t in range(T - 1):
        _state = prior_states[t]
        # 结合信念状态（包含先验前一个动作的特征）和后验状态（传入的事后验状态，包含环境观察的特征）todo
        # 得到预测的动作
        actions = policy.get_action(beliefs[t].detach(), _state.detach())
        # Compute belief (deterministic hidden state)
        # 后验状态结合预测的动作提取特征
        hidden = transition_model.act_fn(transition_model.fc_embed_state_action(torch.cat([_state, actions], dim=1)))
        # 计算得到下一个信念状态
        beliefs[t + 1] = transition_model.rnn(hidden, beliefs[t])
        # Compute state prior by applying transition dynamics
        # 根据信念状态计算先验状态
        hidden = transition_model.act_fn(transition_model.fc_embed_belief_prior(beliefs[t + 1]))
        prior_means[t + 1], _prior_std_dev = torch.chunk(transition_model.fc_state_prior(hidden), 2, dim=1)
        prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + transition_model.min_std_dev
        prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])
    # Return new hidden states
    # imagined_traj = [beliefs, prior_states, prior_means, prior_std_devs]
    imagined_traj = [
        torch.stack(beliefs[1:], dim=0),
        torch.stack(prior_states[1:], dim=0),
        torch.stack(prior_means[1:], dim=0),
        torch.stack(prior_std_devs[1:], dim=0),
    ]
    return imagined_traj


def lambda_return(imged_reward, value_pred, bootstrap, discount=0.99, lambda_=0.95):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    # bootstrap=value_predp[-1]
    # next_value相当于吧value_pred向后移动一位，然后最后一个值用bootstrap填充
    next_values = torch.cat([value_pred[1:], bootstrap[None]], 0)
    # 得到每一个step的discount
    discount_tensor = discount * torch.ones_like(imged_reward)  # pcont
    # 这里有点像bellman公式了，计算出来的是每一步的return，在PPO中可以看到
    # （1 - lambda_）平衡蒙特卡罗回报和时间差分（TD）回报
    inputs = imged_reward + discount_tensor * next_values * (1 - lambda_)
    last = bootstrap
    indices = reversed(range(len(inputs)))
    outputs = []
    # 倒序遍历，这边是在模仿轨迹中计算return
    for index in indices:
        inp, disc = inputs[index], discount_tensor[index]
        last = inp + disc * lambda_ * last
        outputs.append(last)
    # 将顺序反转，变回正序
    outputs = list(reversed(outputs))
    outputs = torch.stack(outputs, 0)
    returns = outputs
    return returns


class ActivateParameters:
    def __init__(self, modules: Iterable[Module]):
        """
        Context manager to locally Activate the gradients.
        example:
        ```
        with ActivateParameters([module]):
            output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            # print(param.requires_grad)
            param.requires_grad = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]


# "get_parameters" and "FreezeParameters" are from the following repo
# https://github.com/juliusfrost/dreamer-pytorch
def get_parameters(modules: Iterable[Module]):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters


class FreezeParameters:
    def __init__(self, modules: Iterable[Module]):
        """
        Context manager to locally freeze gradients.
        In some cases with can speed up computation because gradients aren't calculated for these listed modules.
        example:
        ```
        with FreezeParameters([module]):
            output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]
