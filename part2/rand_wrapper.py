
import gymnasium as gym
import numpy as np
from collections import deque

class RandomizationWrapper(gym.Wrapper):
    """
    Wrapper that applies randomization to the environment.
    """
    def __init__(
        self,
        env,
        mass_range=(1.0, 1.0),
        mode="none",
    ):
        super().__init__(env)

        self.mode = mode
        self.mass_range = mass_range

        # global limits
        self.mass_min_limit, self.mass_max_limit = mass_range

        # Active boundary limits
        if self.mode == "adr":
            # Start with a very narrow range around 1.0 (the source environment default)
            self.mass_min = 1.0
            self.mass_max = 1.0
            self.eval_buffer_size = 30
            self.performance_min = deque(maxlen=self.eval_buffer_size)
            self.performance_max = deque(maxlen=self.eval_buffer_size)
            self.adr_delta = 0.05
            self.t_high = 0.80
            self.t_low = 0.50
            self.p_eval = 0.5
        else:
            self.mass_min = self.mass_min_limit
            self.mass_max = self.mass_max_limit

        self.last_sample_type = "none"

    # -----------------------
    # Mass Sampling
    # -----------------------

    def _sample_mass(self):
        if self.mode == "none":
            self.last_sample_type = "none"
            return None
        elif self.mode == "udr":
            self.last_sample_type = "uniform"
            return np.random.uniform(self.mass_min_limit, self.mass_max_limit)
        elif self.mode == "adr":
            # Decide whether to evaluate a boundary or sample uniformly
            if np.random.rand() < self.p_eval:
                # Evaluate boundary
                if np.random.rand() < 0.5:
                    self.last_sample_type = "min"
                    return self.mass_min
                else:
                    self.last_sample_type = "max"
                    return self.mass_max
            else:
                self.last_sample_type = "uniform"
                return np.random.uniform(self.mass_min, self.mass_max)
        else:
            raise NotImplementedError(f"Sampling strategy '{self.mode}' is not implemented yet.")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        if self.mode == "adr" and done:
            # Check success metric
            success = float(info.get("is_success", 0.0))

            if self.last_sample_type == "min":
                self.performance_min.append(success)
                if len(self.performance_min) >= self.eval_buffer_size:
                    mean_perf = np.mean(self.performance_min)
                    if mean_perf >= self.t_high:
                        # Expand min boundary down (decrease mass_min)
                        old_min = self.mass_min
                        self.mass_min = max(self.mass_min - self.adr_delta, self.mass_min_limit)
                        print(f"[ADR] Expanded mass_min: {old_min:.2f} -> {self.mass_min:.2f} (perf: {mean_perf:.2%})")
                    elif mean_perf < self.t_low:
                        # Shrink min boundary up (increase mass_min towards 1.0)
                        old_min = self.mass_min
                        self.mass_min = min(self.mass_min + self.adr_delta, 1.0)
                        print(f"[ADR] Shrunk mass_min: {old_min:.2f} -> {self.mass_min:.2f} (perf: {mean_perf:.2%})")
                    self.performance_min.clear()

            elif self.last_sample_type == "max":
                self.performance_max.append(success)
                if len(self.performance_max) >= self.eval_buffer_size:
                    mean_perf = np.mean(self.performance_max)
                    if mean_perf >= self.t_high:
                        # Expand max boundary up (increase mass_max)
                        old_max = self.mass_max
                        self.mass_max = min(self.mass_max + self.adr_delta, self.mass_max_limit)
                        print(f"[ADR] Expanded mass_max: {old_max:.2f} -> {self.mass_max:.2f} (perf: {mean_perf:.2%})")
                    elif mean_perf < self.t_low:
                        # Shrink max boundary down (decrease mass_max towards 1.0)
                        old_max = self.mass_max
                        self.mass_max = max(self.mass_max - self.adr_delta, 1.0)
                        print(f"[ADR] Shrunk mass_max: {old_max:.2f} -> {self.mass_max:.2f} (perf: {mean_perf:.2%})")
                    self.performance_max.clear()

        return obs, reward, terminated, truncated, info

    # -----------------------
    # Reset
    # -----------------------

    def reset(self, **kwargs):
        new_mass = self._sample_mass()

        if new_mass is not None:
            sim = self.env.unwrapped.task.sim
            object_body_id = sim._bodies_idx["object"]

            sim.physics_client.changeDynamics(
                bodyUniqueId=object_body_id,
                linkIndex=-1,
                mass=float(new_mass),
            )

            print(
                f"[{self.mode}] mass={new_mass:.2f} "
                f"range=[{self.mass_min:.2f},{self.mass_max:.2f}] "
                f"type={self.last_sample_type}"
            )

        return super().reset(**kwargs)

