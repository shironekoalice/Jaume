"""Fetch small upstream source documents; never download dataset payloads."""
import concurrent.futures
import argparse
import json
from pathlib import Path
import urllib.request

SOURCES = {
    "minimind_model": "https://raw.githubusercontent.com/jingyaogong/minimind-o/master/model/model_omni.py",
    "minimind_trainer": "https://raw.githubusercontent.com/jingyaogong/minimind-o/master/trainer/train_sft_omni.py",
    "emilia_card": "https://huggingface.co/datasets/amphion/Emilia-Dataset/raw/main/README.md",
    "wenet_card": "https://huggingface.co/datasets/Wenetspeech4TTS/WenetSpeech4TTS/raw/main/README.md",
    "minimind_readme": "https://raw.githubusercontent.com/jingyaogong/minimind-o/master/README.md",
    "minimind_tree": "https://api.github.com/repos/jingyaogong/minimind-o/git/trees/master?recursive=1",
    "emilia": "https://raw.githubusercontent.com/open-mmlab/Amphion/main/preprocessors/Emilia/README.md",
    "wenetspeech4tts": "https://raw.githubusercontent.com/wenet-e2e/WenetSpeech4TTS/main/README.md",
    "libritts_r": "https://www.openslr.org/141/",
    "aishell3": "https://www.openslr.org/93/",
    "sensevoice": "https://raw.githubusercontent.com/FunAudioLLM/SenseVoice/main/README.md",
}

def fetch(item):
    name, url = item
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Jaume-research"})
        with urllib.request.urlopen(req, timeout=25) as response:
            body = response.read(2_000_000).decode("utf-8")
        dest = Path("logs/omni-sources")
        dest.mkdir(parents=True, exist_ok=True)
        (dest / (name + ".txt")).write_text(body, encoding="utf-8")
        if name.endswith("tree"):
            paths = [x["path"] for x in json.loads(body).get("tree", [])]
            excerpt = "\n".join(p for p in paths if p.endswith(".py"))
        else:
            terms = ("license", "hour", "小时", "100,000", "data", "dataset", "512", "bridge")
            excerpt = "\n".join(line[:1200] for line in body.splitlines() if any(t in line.lower() for t in terms))[:18000]
        return {"name": name, "url": url, "excerpt": excerpt}
    except Exception as exc:
        return {"name": name, "url": url, "error": str(exc)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", choices=list(SOURCES))
    args = parser.parse_args()
    selected = {name: SOURCES[name] for name in args.names} if args.names else SOURCES
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
        for result in pool.map(fetch, selected.items()):
            print(json.dumps({k: v for k, v in result.items() if k != "excerpt"}, ensure_ascii=True))
