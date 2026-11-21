"""Consistency check: every number in the reports must be backed by a json."""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
DOCS = HERE.parent.parent / "docs"
J = lambda n: json.load(open(HERE / n, encoding="utf-8"))

checks = []


def chk(name, cond):
    checks.append((name, bool(cond)))


# ---- prototype (game + synthetic UI) -------------------------------------
fd = J("final_detector_result.json")
chk("final detector closed-loop min = 1.0", fd["closed_loop"]["min"] == 1.0)
chk("final detector agent min = 1.0", fd["agent"]["min"] == 1.0)

hy = J("hybrid_result.json")
chk("hybrid every=16 success 1.0", hy["rows"][-1]["success"] == 1.0)
chk("hybrid every=16 Hz > 1000", hy["rows"][-1]["effective_hz"] > 1000)

bg = J("browser_ui_result.json")
chk("browser grounding = 1.0 over 20", bg["accuracy"] == 1.0 and bg["n"] == 20)

bl = J("browser_loop_result.json")
chk("browser loop = 1.0 over 6", bl["accuracy"] == 1.0 and bl["n"] == 6)

um = J("ui_multiseed_result.json")
chk("ui full min = 1.0", um["full"]["min"] == 1.0)
chk("ui no_box min = 0.5387", abs(um["no_box"]["min"] - 0.5387) < 1e-4)
chk("ui text_only ~ random", abs(um["text_only"]["mean"] - 1.0 / 6) < 0.02)

ta = J("trunk_attribution.json")
chk("trunk D min = 0.0213", abs(ta["D_coord_no_proj"]["agent_min"] - 0.0213) < 1e-4)
stable = ["A_spatialCNN_coord_proj", "B_conv_only", "C_conv_proj256"]
chk("trunk A/B/C min = 1.0", all(ta[k]["agent_min"] == 1.0 for k in stable))

hi = J("head_init_result.json")
chk("head scale 10 min = 0.0192", abs(hi["scale_10.0"]["agent_min"] - 0.0192) < 1e-4)
chk("head scale 1 min = 1.0", hi["scale_1.0"]["agent_min"] == 1.0)

ss = J("spatial_softmax_result.json")
chk("ssm B mean = 0.0607", abs(ss["B_spatial_softmax"]["agent_mean"] - 0.0607) < 1e-3)
chk("ssm A min = 1.0", ss["A_classification"]["agent_min"] == 1.0)

gn = J("maze_generalisation.json")
chk("gen D0 = 0.890", abs(gn["D0_train_dist"]["success"] - 0.890) < 1e-3)
chk("gen D3 = 0.010", abs(gn["D3_grid12"]["success"] - 0.010) < 1e-3)
chk("gen D5 = 0.875", abs(gn["D5_shift_palette"]["success"] - 0.875) < 1e-3)

# ---- backbone selection --------------------------------------------------
ps = J("probe_siglip.json")["siglip2_base"]
chk("siglip probe AUROC = 0.8991", abs(ps["auroc"] - 0.8991) < 1e-4)
chk("siglip probe top1 = 0.8500", abs(ps["top1"] - 0.8500) < 1e-4)
chk("siglip probe is held-out", ps["n_test"] > 0 and ps["n_train"] > 0)

pq3 = J("probe_qwen3vl.json")["qwen3vl_2b"]
chk("qwen3vl probe AUROC = 0.9472", abs(pq3["auroc"] - 0.9472) < 1e-4)
chk("qwen3vl probe top1 = 0.8056", abs(pq3["top1"] - 0.8056) < 1e-4)

pq2 = J("probe_qwen2vl.json")["qwen2vl_2b"]
chk("qwen2vl probe AUROC = 0.9046", abs(pq2["auroc"] - 0.9046) < 1e-4)
chk("qwen2vl probe top1 = 0.6667", abs(pq2["top1"] - 0.6667) < 1e-4)

hc = J("head_comparison.json")
chk("random-distractor: linear probe 0.8889", abs(hc["P_linear_probe"]["last_top1"] - 0.8889) < 1e-3)
chk("random-distractor: gated 1.0000", hc["G_gated"]["last_top1"] == 1.0)
chk("random-distractor: no-instruction control ALSO 1.0 (the broken benchmark)",
    hc["N_gated_no_instruction"]["last_top1"] == 1.0)

rd = J("real_distractor_result.json")
chk("real-distractor: linear probe 0.1846 (best)", abs(rd["P_linear_probe"]["best_top1"] - 0.1846) < 1e-3)
chk("real-distractor: gated 0.6483 (best)", abs(rd["G_gated"]["best_top1"] - 0.6483) < 1e-3)
chk("real-distractor: no-instruction control 0.1827 (best)",
    abs(rd["N_gated_no_instruction"]["best_top1"] - 0.1827) < 1e-3)
chk("real-distractor: control is at the floor (0.1667)",
    abs(rd["N_gated_no_instruction"]["best_top1"] - 1.0 / 6) < 0.03)
chk("real-distractor: gated beats control by > 0.4",
    rd["G_gated"]["best_top1"] - rd["N_gated_no_instruction"]["best_top1"] > 0.4)

il = J("inference_layouts.json")
q = list(il.values())[0]
chk("L1 full prompt latency ~9000 ms", 8000 < q["L1_full_prompt"]["latency_ms"] < 10000)
chk("L1 top1 = 0.48", abs(q["L1_full_prompt"]["top1"] - 0.48) < 0.03)
chk("L2 regression acc = 0.14", abs(q["L2_screen_only_regression"]["regression_acc"] - 0.14) < 0.03)

# ---- real DOM -----------------------------------------------------------
bgd = J("browser_gated_result.json")["summary"]
chk("real DOM: gated head top1 = 0.0", bgd["top1"] == 0.0)
chk("real DOM: no-instruction control ALSO 0.0", bgd["top1_no_instruction"] == 0.0)

sco = J("single_crop_ocr.json")["summary"]
bdb = J("browser_dom_qwen3vl.json")["summary"]
chk("real DOM: Qwen3-VL multi-crop prompt top1 = 0.2857",
    abs(bdb["top1"] - 0.2857) < 1e-3)
chk("real DOM: Qwen3-VL multi-crop no-instruction = 0.1429 (floor)",
    abs(bdb["top1_no_instruction"] - 0.1429) < 1e-3)
chk("real DOM: Qwen3-VL multi-crop latency ~16.5 s",
    15000 < bdb["latency_ms"] < 18000)
chk("real DOM: multi-crop is far below the single-crop result",
    bdb["top1"] < 0.5 * sco["task_top1"])

ts = J("twostage_dom_result.json")["summary"]
chk("two-stage (TrOCR) top1 = 0.1818", abs(ts["top1"] - 0.1818) < 1e-3)

chk("single-crop OCR: task top1 = 1.0000", sco["task_top1"] == 1.0)
chk("single-crop OCR: OCR exact = 0.875", abs(sco["ocr_exact"] - 0.875) < 1e-3)
chk("single-crop OCR: beats the floor by a lot",
    sco["task_top1"] - sco["floor"] > 0.5)
chk("single-crop OCR: per-crop latency ~860 ms",
    700 < sco["ms_per_crop"] < 1000)

for name, ok in checks:
    print(("  OK   " if ok else "  FAIL ") + name)
print()
print("ALL PASS" if all(o for _, o in checks) else "SOME FAILED")
