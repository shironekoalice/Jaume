import io
p = "code/repo/scripts/train_vlm_probe.py"
s = io.open(p, encoding="utf-8").read()
a = '    ap.add_argument("--unfreeze-layers", type=int, default=0,'
ins = ('    ap.add_argument("--task", choices=["count", "color_count"],\n'
       '                    default="count")\n'
       '    ap.add_argument("--visual-backbone",\n'
       '                    choices=["cnn", "resnet18", "resnet34"],\n'
       '                    default="cnn")\n'
       '    ap.add_argument("--visual-pretrained", action="store_true")\n')
assert a in s, "anchor missing"
s = s.replace(a, ins + a, 1)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("args added")
