import multiprocessing as mp
import numpy as np

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                env.step(data)
                remote.send(True)
            elif cmd == 'reset':
                obs_dict = env.reset(seed=data)
                remote.send(obs_dict)
            elif cmd == 'observe':
                obs = env.observe(data)
                remote.send(obs)
            elif cmd == 'agent_selection':
                remote.send(env.agent_selection)
            elif cmd == 'terminations':
                remote.send(env.terminations)
            elif cmd == 'infos':
                remote.send(env.infos)
            elif cmd == 'rewards':
                remote.send(env.rewards)
            elif cmd == 'cumulative_rewards':
                # BUG FIX: env.rewards is zeroed after each step by _accumulate_rewards.
                # _cumulative_rewards holds the correct per-episode accumulated rewards.
                remote.send(env._cumulative_rewards)
            elif cmd == 'state':
                remote.send(env.state())
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError(f"Command {cmd} is not implemented.")
        except EOFError:
            break

class CloudVectorEnv:
    """A minimal SubprocVecEnv specifically tailored for PettingZoo AEC OmiEnv."""
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)

        import os
        # BUG FIX: use 'forkserver' context on Linux to avoid CUDA/NCCL deadlocks,
        # but Windows doesn't support 'fork' or 'forkserver' — it only supports 'spawn'.
        if os.name == 'nt':
            ctx = mp.get_context("spawn")
        else:
            ctx = mp.get_context("forkserver")
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, env_fn))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]

        for p in self.ps:
            p.daemon = True  # Ensures processes close when main process closes
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def _send_to_active(self, cmd, data_list, env_indices):
        active_remotes = [self.remotes[i] for i in env_indices]
        for remote, data in zip(active_remotes, data_list):
            remote.send((cmd, data))
        
        results = []
        for remote in active_remotes:
            res = remote.recv()
            results.append(res)
        return results

    def step(self, actions: list, env_indices: list = None):
        if env_indices is None: env_indices = list(range(self.num_envs))
        return self._send_to_active('step', actions, env_indices)

    def reset(self, seeds: list = None, env_indices: list = None):
        if env_indices is None: env_indices = list(range(self.num_envs))
        if seeds is None: seeds = [None]*len(env_indices)
        return self._send_to_active('reset', seeds, env_indices)

    def observe(self, agent_names: list, env_indices: list = None):
        if env_indices is None: env_indices = list(range(self.num_envs))
        return self._send_to_active('observe', agent_names, env_indices)

    def agent_selection(self, env_indices: list = None):
        if env_indices is None: env_indices = list(range(self.num_envs))
        return self._send_to_active('agent_selection', [None]*len(env_indices), env_indices)

    def get_terminations(self, env_indices: list = None):
        if env_indices is None: env_indices = list(range(self.num_envs))
        return self._send_to_active('terminations', [None]*len(env_indices), env_indices)

    def get_infos(self, env_indices: list = None):
        if env_indices is None: env_indices = list(range(self.num_envs))
        return self._send_to_active('infos', [None]*len(env_indices), env_indices)

    def get_rewards(self, env_indices: list = None):
        if env_indices is None: env_indices = list(range(self.num_envs))
        return self._send_to_active('rewards', [None]*len(env_indices), env_indices)

    def get_cumulative_rewards(self, env_indices: list = None):
        # BUG FIX (Bug 1): returns _cumulative_rewards, not rewards.
        # env.rewards is zeroed inside env.step() → _accumulate_rewards(), so
        # get_rewards() always returns 0.0 after a step. _cumulative_rewards
        # holds the correctly accumulated episode rewards.
        if env_indices is None: env_indices = list(range(self.num_envs))
        return self._send_to_active('cumulative_rewards', [None]*len(env_indices), env_indices)

    def get_state(self, env_indices: list = None):
        if env_indices is None: env_indices = list(range(self.num_envs))
        return self._send_to_active('state', [None]*len(env_indices), env_indices)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
