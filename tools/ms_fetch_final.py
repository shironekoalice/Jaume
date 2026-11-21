from modelscope.hub.snapshot_download import snapshot_download
p = snapshot_download(
    "LightNeko/Jaume",
    allow_patterns=["output/350m-causal-mlm/final/*"],
    cache_dir="D:/项目/Jaume/.mscache",
)
print("SNAP", p)
