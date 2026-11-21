"""Where does the OCR/semantics conflict live?

Three single-task runs from the S2 checkpoint, 2 epochs each:
  A) OCR only      -> can text_locate recover? (conflict is interference)
  B) semantics only-> does VIS improve further? (capacity is not the limit)
  C) both, 2:1 OCR -> does task weighting keep both?

If A recovers text_locate quickly, the capability is intact and the problem is
gradient interference, not capacity. That distinction decides the fix.
"""
import argparse, os, sys, random, string, time
import torch, torch.nn as nn, torch.nn.functional as F

if "torchaudio" not in sys.modules:
    import importlib.machinery as _im, types
    def _stub(n):
        m = types.ModuleType(n); m.__version__ = "0"
        m.__spec__ = _im.ModuleSpec(n, loader=None); m.__path__ = []
        m.__getattr__ = lambda a: None; sys.modules[n] = m; return m
    _stub("torchaudio")
    for _s in ("torchaudio.transforms","torchaudio.functional",
               "torchaudio.models","torchaudio.utils"):
        _stub(_s)

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
REPO = os.path.join(PROJECT_ROOT, "code", "repo")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jaume.device import resolve_device, seed_all
from ocr_suite import build as build_ocr, OCR_FAMILIES
from vision_suite import build as build_vis
VIS_FAMILIES = ["attr","count","spatial","compare","scene","multi_attr"]

CHARS = list(string.digits)+list(string.ascii_letters)+["$"," ",".","-"]
C2I = {c:i for i,c in enumerate(CHARS)}
PAD_C = len(CHARS); N_C = len(CHARS)+1; MAX_LEN = 8

def enc_ans(a):
    ids=[C2I[c] for c in a if c in C2I][:MAX_LEN]
    return ids+[PAD_C]*(MAX_LEN-len(ids))

class Dual(nn.Module):
    def __init__(self,d,sig,tro):
        super().__init__()
        from transformers import SiglipModel, VisionEncoderDecoderModel
        f=SiglipModel.from_pretrained(sig,dtype=torch.float32); self.sem=f.vision_model
        self.proj_sem=nn.Sequential(nn.LayerNorm(self.sem.config.hidden_size),
            nn.Linear(self.sem.config.hidden_size,d),nn.GELU(),nn.Linear(d,d))
        t=VisionEncoderDecoderModel.from_pretrained(tro,dtype=torch.float32); self.ocr=t.encoder
        self.proj_ocr=nn.Sequential(nn.LayerNorm(self.ocr.config.hidden_size),
            nn.Linear(self.ocr.config.hidden_size,d),nn.GELU(),nn.Linear(d,d))
        for p in self.sem.parameters(): p.requires_grad_(False)
        for p in self.ocr.parameters(): p.requires_grad_(False)
    def forward(self,sp,op):
        return (self.proj_sem(self.sem(pixel_values=sp).last_hidden_state),
                self.proj_ocr(self.ocr(pixel_values=op).last_hidden_state))

def prep(imgs,size):
    import torchvision.transforms.functional as TF
    xs=[]
    for im in imgs:
        x=TF.to_tensor(im.convert("RGB")); x=TF.resize(x,[size,size],antialias=True)
        xs.append((x-0.5)/0.5)
    return torch.stack(xs)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True); ap.add_argument("--siglip-path",required=True)
    ap.add_argument("--trocr-path",required=True); ap.add_argument("--s2-ckpt",required=True)
    ap.add_argument("--mode",choices=["ocr","vis","both21"],required=True)
    ap.add_argument("--epochs",type=int,default=2); ap.add_argument("--batch-size",type=int,default=8)
    ap.add_argument("--device",default=None)
    a=ap.parse_args()
    dev=resolve_device(a.device); seed_all(dev,0)
    from jaume.modeling.config import JaumeConfig
    from jaume.modeling.embedding import JaumeEmbeddingModel
    from jaume.modeling.checkpoint import load_state_dict
    from jaume.data.dataset import load_tokenizer
    cfg=JaumeConfig.load(os.path.join(a.model,"config.json"))
    tok=load_tokenizer(os.path.join(a.model,"tokenizer")); tok.add_tokens(["[MASK]"])
    cfg.vocab_size=len(tok); cfg.causal=True
    enc=JaumeEmbeddingModel(cfg).to(dev)
    pre=load_state_dict(os.path.join(a.model,"pytorch_model.bin"))
    own=enc.state_dict(); merged={}
    for k in own: merged[k]=pre[k] if (k in pre and pre[k].shape==own[k].shape) else own[k]
    enc.load_state_dict(merged,strict=True)
    for p in enc.parameters(): p.requires_grad_(False)
    for layer in list(enc.backbone.layers)[-4:]:
        for p in layer.parameters(): p.requires_grad_(True)
    for p in enc.backbone.norm.parameters(): p.requires_grad_(True)

    d=cfg.hidden_size
    tower=Dual(d,a.siglip_path,a.trocr_path).to(dev)
    chead=nn.Linear(d,N_C).to(dev)
    ck=torch.load(a.s2_ckpt,map_location="cpu",weights_only=False)
    tower.load_state_dict(ck["tower"],strict=False); chead.load_state_dict(ck["char_head"])
    print("[diag] loaded S2, mode=%s" % a.mode, flush=True)

    otr=build_ocr(2000,seed=31); ova=build_ocr(300,seed=909)
    vtr=build_vis(1600,seed=11); vva=build_vis(300,seed=777)
    qt={}
    for _,_,_,_,_,q in otr+ova+vtr+vva:
        if q not in qt:
            qt[q]=torch.tensor(tok(q,add_special_tokens=False)["input_ids"][:32],
                               dtype=torch.long,device=dev)

    def run_text(b):
        sp=prep([x[0] for x in b],224).to(dev); op=prep([x[0] for x in b],384).to(dev)
        s,o=tower(sp,op); B=len(b)
        qs=[x[5] for x in b]; ans=[x[3] for x in b]
        ql=max(len(qt[q]) for q in qs)
        aid=[torch.tensor(enc_ans(x),device=dev) for x in ans]
        qids=torch.zeros(B,ql,dtype=torch.long,device=dev); qm=torch.zeros_like(qids)
        for i,q in enumerate(qs):
            n=len(qt[q]); qids[i,:n]=qt[q]; qm[i,:n]=1
        ain=torch.full((B,MAX_LEN),PAD_C,dtype=torch.long,device=dev)
        for i,ids in enumerate(aid): ain[i,1:]=ids[:-1]
        emb=enc.backbone.embed_tokens(torch.cat([qids,ain],1))
        x=torch.cat([s,o,emb],1)
        mask=torch.cat([torch.ones(B,s.shape[1],dtype=torch.long,device=dev),
                        torch.ones(B,o.shape[1],dtype=torch.long,device=dev),qm,
                        torch.ones(B,MAX_LEN,dtype=torch.long,device=dev)],1)
        h=x
        for L in enc.backbone.layers: h,_=L(h,mask)
        h=enc.backbone.norm(h)
        base=s.shape[1]+o.shape[1]+ql
        logits=chead(h[:,base:base+MAX_LEN,:].float())
        tgt=torch.stack(aid)
        loss=F.cross_entropy(logits.reshape(-1,N_C),tgt.reshape(-1))
        pred=logits.argmax(-1)
        ok=[all(pred[i,j].item()==tgt[i,j].item() for j in range(len(enc_ans(ans[i]))))
            for i in range(B)]
        return loss,ok

    def ev():
        tower.eval(); chead.eval(); enc.eval()
        per={f:[0,0] for f in OCR_FAMILIES+VIS_FAMILIES}
        with torch.no_grad():
            for src in (ova,vva):
                for i in range(0,len(src)-4+1,4):
                    b=src[i:i+4]; _,ok=run_text(b)
                    for j,o_ in enumerate(ok):
                        per[b[j][1]][1]+=1; per[b[j][1]][0]+=int(o_)
        return per

    ps=list(tower.proj_sem.parameters())+list(tower.proj_ocr.parameters())+list(chead.parameters())
    ep_=[p for p in enc.parameters() if p.requires_grad]
    opt=torch.optim.AdamW([{"params":ps,"lr":5e-4},{"params":ep_,"lr":5e-5}],weight_decay=0.01)
    allp=ps+ep_

    for e in range(a.epochs):
        tower.train(); chead.train(); enc.train()
        tot=nb=0; t0=time.time()
        if a.mode=="ocr": src=otr
        elif a.mode=="vis": src=vtr
        else: src=otr+otr+vtr   # 2:1 OCR weighting
        perm=list(range(len(src))); random.Random(e).shuffle(perm)
        for i in range(0,len(src)-a.batch_size+1,a.batch_size):
            b=[src[j] for j in perm[i:i+a.batch_size]]
            loss,_=run_text(b)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(allp,1.0); opt.step()
            tot+=float(loss); nb+=1
        per=ev()
        om=sum(per[f][0]/max(per[f][1],1) for f in OCR_FAMILIES)/3
        vm=sum(per[f][0]/max(per[f][1],1) for f in VIS_FAMILIES)/6
        print("[diag] %s ep%d loss %.3f | OCR %.3f (locate %.2f) | VIS %.3f (%.0fs)"
              % (a.mode,e+1,tot/max(nb,1),om,
                 per["text_locate"][0]/max(per["text_locate"][1],1),vm,time.time()-t0),
              flush=True)

if __name__=="__main__":
    main()
