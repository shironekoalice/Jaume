import io
p = "code/repo/scripts/train_vlm_probe.py"
s = io.open(p, encoding="utf-8").read()

backbone_cls = (
    "class ResNetStem(nn.Module):\n"
    "    def __init__(self, d_out, arch='resnet18', pretrained=False):\n"
    "        super().__init__()\n"
    "        import torchvision.models as tvm\n"
    "        net = getattr(tvm, arch)(weights='DEFAULT' if pretrained else None)\n"
    "        self.body = nn.Sequential(\n"
    "            net.conv1, net.bn1, net.relu, net.maxpool,\n"
    "            net.layer1, net.layer2, net.layer3, net.layer4)\n"
    "        c = net.fc.in_features\n"
    "        self.proj = nn.Linear(c, d_out)\n"
    "\n"
    "    def forward(self, x):\n"
    "        h = self.body(x)\n"
    "        B, C, H, W = h.shape\n"
    "        t = h.flatten(2).transpose(1, 2)\n"
    "        return self.proj(t)\n"
    "\n"
    "\n"
)

anchor = "class PatchEncoder(nn.Module):"
assert anchor in s and "class ResNetStem" not in s
s = s.replace(anchor, backbone_cls + anchor, 1)

s = s.replace(
    "    ap.add_argument('--unfreeze-layers', type=int, default=0,",
    "    ap.add_argument('--visual-backbone',\n"
    "                    choices=['cnn', 'resnet18', 'resnet34'],\n"
    "                    default='cnn')\n"
    "    ap.add_argument('--visual-pretrained', action='store_true')\n"
    "    ap.add_argument('--unfreeze-layers', type=int, default=0,")

s = s.replace(
    "    vis = PatchEncoder(d).to(device)",
    "    if args.visual_backbone == 'cnn':\n"
    "        vis = PatchEncoder(d).to(device)\n"
    "    else:\n"
    "        vis = ResNetStem(d, args.visual_backbone,\n"
    "                         args.visual_pretrained).to(device)\n"
    "    print('[vlm] visual backbone=%s pretrained=%s' % (\n"
    "        args.visual_backbone, args.visual_pretrained), flush=True)")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched")
