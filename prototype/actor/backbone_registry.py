"""Registry of candidate backbones for the actor.

Each entry builds a scorer with

    score(image, instruction, candidates) -> np.ndarray of per-candidate scores

Three kinds of scorer, all ZERO-SHOT (no training), because that is the fair
way to compare backbones before spending compute:

  SigLIP2      crop embedding . instruction embedding -> cosine similarity.
               "Which crop does this sentence describe?" No LM at all.

  VLM (Qwen/InternVL)   the screenshot and the candidate crops are shown
               together, the candidates are lettered, and the answer is read
               from the LOGIT of each option letter at the first answer
               position. ONE forward pass, no autoregressive decoding —
               exactly the actor's action space.

  Florence-2   a task-prompted grounding specialist. Its native output is a
               box, not a ranking, so it is scored in ITS OWN action space
               and the result is converted into candidate scores by box
               overlap. This makes the comparison honest: Florence-2 is
               measured at what it is good at, and still reported in the
               candidate-scoring metric so the numbers are comparable.

Weights come from ModelScope (HuggingFace is unreachable from this host);
`fetch_models.py` puts them in `models/<namespace>__<name>/`.
"""
from __future__ import annotations

import gc
import os

import numpy as np
import torch
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.environ.get("ACTOR_MODELS_DIR", os.path.join(BASE, "models"))


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


DEV = _device()


def local_path(model_id: str) -> str:
    """ModelScope layout: models/<namespace>__<name>/."""
    p = os.path.join(MODELS_DIR, model_id.replace("/", "__"))
    return p if os.path.isdir(p) else model_id


def crop_image(image, box, size=224):
    x0, y0, x1, y1 = [int(v) for v in box]
    x0 = max(0, min(image.width - 1, x0))
    y0 = max(0, min(image.height - 1, y0))
    x1 = max(x0 + 1, min(image.width, x1))
    y1 = max(y0 + 1, min(image.height, y1))
    return image.crop((x0, y0, x1, y1)).resize((size, size), Image.BICUBIC)


def _free():
    gc.collect()
    if DEV == "xpu":
        torch.xpu.empty_cache()
    elif DEV == "cuda":
        torch.cuda.empty_cache()


class SigLIPScorer:
    NAME = "siglip2_base"

    def __init__(self, model_id="google/siglip2-base-patch16-224", crop_size=224):
        from transformers import AutoModel, AutoProcessor

        self.crop_size = crop_size
        self.proc = AutoProcessor.from_pretrained(local_path(model_id))
        self.model = AutoModel.from_pretrained(local_path(model_id)).to(DEV).eval()

    @torch.no_grad()
    def score(self, image, instruction, candidates):
        crops = [crop_image(image, b, self.crop_size) for b in candidates]
        img_in = {k: v.to(DEV) for k, v in self.proc(images=crops, return_tensors="pt").items()}
        img_emb = self.model.get_image_features(**img_in) \
            if hasattr(self.model, "get_image_features") else self.model(**img_in).pooler_output
        if img_emb.dim() == 3:
            img_emb = img_emb.mean(dim=1)
        txt_in = {k: v.to(DEV) for k, v in
                  self.proc(text=[instruction], return_tensors="pt",
                            padding=True, truncation=True, max_length=64).items()}
        if hasattr(self.model, "get_text_features"):
            txt_emb = self.model.get_text_features(**txt_in)
        else:
            txt_emb = self.model(**txt_in).pooler_output
        if txt_emb.dim() == 3:
            txt_emb = txt_emb.mean(dim=1)
        a = torch.nn.functional.normalize(img_emb, dim=-1)
        b = torch.nn.functional.normalize(txt_emb, dim=-1)
        return (a @ b.T).squeeze(-1).float().cpu().numpy()

    @torch.no_grad()
    def embed(self, crops, texts, kind=None):
        """Frozen JOINT features for the linear probe.

        For a dual tower, the crop embedding and the instruction embedding
        are CONCATENATED (after normalisation). The probe then learns how to
        match them — which is exactly the grounding question, and it lets the
        probe exploit the complementarity that the raw cosine score misses
        (the zero-shot cosine of this tower scored 0.083 on ScreenSpot,
        BELOW the 0.167 random floor, while a probe on the same features
        does far better — the information is there, the zero-shot readout
        was wrong).
        """
        img_in = {k: v.to(DEV) for k, v in self.proc(images=crops, return_tensors="pt").items()}
        img_emb = self.model.get_image_features(**img_in) \
            if hasattr(self.model, "get_image_features") else self.model(**img_in).pooler_output
        if img_emb.dim() == 3:
            img_emb = img_emb.mean(dim=1)
        txt_in = {k: v.to(DEV) for k, v in
                  self.proc(text=texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=64).items()}
        txt_emb = self.model.get_text_features(**txt_in) \
            if hasattr(self.model, "get_text_features") else self.model(**txt_in).pooler_output
        if txt_emb.dim() == 3:
            txt_emb = txt_emb.mean(dim=1)
        a = torch.nn.functional.normalize(img_emb.float(), dim=-1)
        b = torch.nn.functional.normalize(txt_emb.float(), dim=-1)
        return dict(crop=torch.cat([a, b], dim=-1).cpu(), text=b.cpu())

    def unload(self):
        del self.model
        _free()


class VLMScorer:
    """One forward; read the option-letter logits.

    Works with any Qwen-VL / InternVL chat template that accepts a list of
    images. The prompt lists the candidates as letters; the answer is the
    argmax over the logits of the letter tokens at the last position.
    """

    LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(self, model_id, crop_size=224, cls=None):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.model_id = model_id
        self.crop_size = crop_size
        p = local_path(model_id)
        # InternVL2 ships custom modelling code; Qwen-VL does not, but
        # trust_remote_code is harmless for it.
        self.proc = AutoProcessor.from_pretrained(p, trust_remote_code=True)
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                p, dtype=torch.float32, trust_remote_code=True).to(DEV).eval()
        except Exception:
            from transformers import AutoModelForVision2Seq

            self.model = AutoModelForVision2Seq.from_pretrained(
                p, dtype=torch.float32, trust_remote_code=True).to(DEV).eval()

    def _messages(self, image, instruction, candidates):
        crops = [crop_image(image, b, self.crop_size) for b in candidates]
        letters = "\n".join(f"{self.LETTERS[i]}. candidate {i+1}" for i in range(len(crops)))
        text = (f"The first image is the full screen. The next {len(crops)} images are "
                f"candidate crops, in order, labelled A to {self.LETTERS[len(crops)-1]}.\n"
                f"{letters}\n"
                f"Instruction: {instruction}\n"
                f"Which candidate crop does the instruction refer to? Answer with one letter.")
        content = [{"type": "image", "image": image}]
        for c in crops:
            content.append({"type": "image", "image": c})
        content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}], [image] + crops

    @torch.no_grad()
    def score(self, image, instruction, candidates):
        msgs, images = self._messages(image, instruction, candidates)
        prompt = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[prompt], images=images, return_tensors="pt")
        inputs = {k: (v.to(DEV) if hasattr(v, "to") else v) for k, v in inputs.items()}
        logits = self.model(**inputs).logits[0, -1].float()
        tok = self.proc.tokenizer
        out = []
        for i in range(len(candidates)):
            ids = tok.encode(self.LETTERS[i], add_special_tokens=False)
            out.append(float(logits[ids[-1]]))
        return np.array(out)

    @torch.no_grad()
    def embed(self, crops, texts, kind=None):
        """Frozen features for the linear probe.

        For a VLM the crop is encoded on its own and the instruction is
        appended as text in the same sequence, and the final hidden state is
        taken. This gives a (crop, instruction) JOINT feature, which is what
        a grounding probe needs: the representation must already fuse "what
        this crop looks like" with "what was asked".

        The crops are encoded ONE PER FORWARD (a VLM with many images in one
        prompt does not give per-image hidden states), but the image
        ENCODER is run on all crops of a batch at once where possible, so
        the cost is one LLM pass per crop and the vision tower is batched.
        """
        outs = []
        B = 8
        for s in range(0, len(crops), B):
            chunk_c = crops[s:s + B]
            chunk_t = texts[s:s + B]
            for c, t in zip(chunk_c, chunk_t):
                msgs = [{"role": "user", "content": [
                    {"type": "image", "image": c},
                    {"type": "text", "text": f"Element: {t}"}]}]
                prompt = self.proc.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
                inputs = self.proc(text=[prompt], images=[c], return_tensors="pt")
                inputs = {k: (v.to(DEV) if hasattr(v, "to") else v) for k, v in inputs.items()}
                out = self.model(**inputs, output_hidden_states=True)
                h = out.hidden_states[-1][0].mean(dim=0)
                outs.append(h.float().cpu())
        return dict(crop=torch.stack(outs), text=torch.zeros(len(outs), 1))

    def unload(self):
        del self.model
        _free()


class FlorenceScorer:
    """Florence-2 answers with a box; convert to candidate scores by IoU.

    Florence-2 is a grounding specialist (230M) trained exactly for this.
    Its native prompt is '<CAPTION_TO_PHRASE_GROUNDING>' + the phrase, and it
    emits a single box. Scoring every candidate would mean K forwards, so
    instead ONE forward predicts the box and the candidates are ranked by IoU
    with it — which is the fair way to use a model whose output space is a
    box.
    """

    def __init__(self, model_id="microsoft/Florence-2-base"):
        from transformers import AutoModelForCausalLM, AutoProcessor

        p = local_path(model_id)
        self.proc = AutoProcessor.from_pretrained(p, trust_remote_code=True)
        # Florence-2 ships remote code written against an older transformers;
        # the default sdpa path touches an attribute that no longer exists.
        # "eager" is the compatible path.
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                p, trust_remote_code=True, dtype=torch.float32,
                attn_implementation="eager").to(DEV).eval()
        except Exception as e:
            raise RuntimeError(f"Florence-2 load failed: {e}") from e

    @torch.no_grad()
    def score(self, image, instruction, candidates):
        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        inputs = self.proc(text=task + instruction, images=image, return_tensors="pt")
        inputs = {k: v.to(DEV) for k, v in inputs.items()}
        # num_beams=1: Florence-2's remote code assumes the OLD transformers
        # cache layout, and beam search touches it (past_key_values[0][0].shape)
        # and crashes on the current version. Greedy decoding avoids the path.
        ids = self.model.generate(**inputs, max_new_tokens=64, num_beams=1, do_sample=False)
        text = self.proc.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self.proc.post_process_generation(
            text, task=task, image_size=(image.width, image.height))[0]
        box = None
        for v in parsed.values():
            if isinstance(v, dict) and "bboxes" in v and len(v["bboxes"]):
                box = tuple(v["bboxes"][0]); break
        if box is None:
            return np.zeros(len(candidates))
        bx = (box[0], box[1], box[0] + box[2], box[1] + box[3])
        ious = []
        for c in candidates:
            cx0, cy0, cx1, cy1 = c
            ix0, iy0 = max(bx[0], cx0), max(bx[1], cy0)
            ix1, iy1 = min(bx[2], cx1), min(bx[3], cy1)
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            a1 = (bx[2] - bx[0]) * (bx[3] - bx[1])
            a2 = (cx1 - cx0) * (cy1 - cy0)
            u = a1 + a2 - inter
            ious.append(inter / u if u > 0 else 0.0)
        return np.array(ious)

    def unload(self):
        del self.model
        _free()


CANDIDATES = {
    "siglip2_base": dict(
        id="google/siglip2-base-patch16-224",
        build=lambda: SigLIPScorer("google/siglip2-base-patch16-224")),
    # Florence-2 is NOT in the set: its remote code targets an older
    # transformers (the cache layout in prepare_inputs_for_generation changed)
    # and it cannot be made to run here without patching vendored code. Its
    # native action space is also a box, not a ranking, so it would not be
    # measured in the actor's action space anyway.
    "internvl2_1b": dict(
        id="OpenGVLab/InternVL2-1B",
        build=lambda: VLMScorer("OpenGVLab/InternVL2-1B")),
    "qwen3vl_2b": dict(
        id="Qwen/Qwen3-VL-2B-Instruct",
        build=lambda: VLMScorer("Qwen/Qwen3-VL-2B-Instruct")),
    "qwen2vl_2b": dict(
        id="Qwen/Qwen2-VL-2B-Instruct",
        build=lambda: VLMScorer("Qwen/Qwen2-VL-2B-Instruct")),
}
