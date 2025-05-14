# export_ppo_deterministic.py
import torch, yaml
from stable_baselines3 import PPO
from duckietown_utils.env import launch_and_wrap_env

MODEL_ZIP  = "/workspace/Duckietown-RL/artifacts/Train_0030_0000/Apr25_03-27-57/best_model.zip"
CFG_YML    = "/workspace/Duckietown-RL/artifacts/Train_0030_0000/Apr25_03-27-57/config_dump_0000.yml"
ONNX_OUT   = "/workspace/Duckietown-RL/exported_model/new_export/real_gray_old.onnx"

cfg = yaml.safe_load(open(CFG_YML))
cfg["env_config"]["mode"] = "inference"
env = launch_and_wrap_env(cfg["env_config"])

model = PPO.load(MODEL_ZIP, env=env)
model.policy.eval().cpu()

# --- kleiner Wrapper, der *immer* deterministic=True nutzt -----------
class DeterministicPolicy(torch.nn.Module):
    def __init__(self, sb3_policy):
        super().__init__()
        self.sb3_policy = sb3_policy
    @torch.no_grad()
    def forward(self, x):
        # _predict() liefert nur die Aktion (ohne log‑prob, ohne states)
        return self.sb3_policy._predict(x, deterministic=True)

det_pol = DeterministicPolicy(model.policy)

dummy_input = torch.zeros(1, *env.observation_space.shape)  # (1,C,H,W)

torch.onnx.export(
    det_pol,
    dummy_input,
    ONNX_OUT,
    opset_version=12,
    input_names = ["input"],
    output_names = ["action"],   # shape (1,1)
    dynamic_axes = {"input":{0:"batch"}, "action":{0:"batch"}},
)
print("✅ deterministisches ONNX gespeichert:", ONNX_OUT)

