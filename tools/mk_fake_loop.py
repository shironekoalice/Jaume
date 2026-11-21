"""Make a throwaway loop-LM checkpoint so probe_loop.py can be exercised."""
import os, sys, torch, json
sys.path.insert(0, "code/repo")
from jaume.modeling.config import JaumeConfig
from jaume.modeling.loop_lm import LoopLM

out = "output/_loop_probe_fake"
os.makedirs(out, exist_ok=True)
cfg = JaumeConfig.load("code/repo/configs/jaume-350m-loop.json")
cfg.num_loops = 1
m = LoopLM(cfg)
sd = {"backbone." + k: v for k, v in m.backbone.state_dict().items()}
sd["lm_head.weight"] = m.lm_head.weight.detach().cpu()
torch.save(sd, os.path.join(out, "pytorch_model.bin"))
cfg.save(os.path.join(out, "config.json"))
print("fake checkpoint ->", out, os.path.getsize(os.path.join(out, "pytorch_model.bin")), "bytes")
