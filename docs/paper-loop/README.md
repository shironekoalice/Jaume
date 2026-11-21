# Depth Saturation Predicts When Looping Cannot Help

Working draft. Evidence lives in  §8.

## One-line claim

Before spending compute on a looped/recurrent transformer, run a depth scan:
if 16L does not clearly beat 8L on the target task, looping is structurally
guaranteed to lose to training a deeper model on more data at equal compute.
