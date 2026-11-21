# JM-S fix experiments, grounded in Slot Attention (Locatello et al. 2020,
# arXiv:2006.15055). Root cause in the v1 run: slots converged to identical
# reads (slot-state cosine 0.87, JS 0.04) because the softmax was over TOKENS
# computed independently per slot -> no competition.
#
#   C1  slot competition (softmax over slots)               - the paper's core fix
#   C2  C1 + 3 iterative refinements + GRU slot update      - the full module
#   C3  C2 + diversity loss on slot states                  - explicit anti-collapse
#   C4  C2 + slot count 16                                  - does capacity help once slots specialise?
#   C5  competition + slot-pool=attention                   - learned slot aggregation
#   C6  competition + 3 iters, NO GRU (plain residual)      - is the GRU necessary?
#
# Every run: same base (350m-mlm), same data/steps/lr/max_length as the
# original sweep, seed 1 (matches B3-latent-s1 for a paired comparison).
$py = 'C:\Users\JingYue\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe'
$env:HF_ENDPOINT = 'https://hf-mirror.com'
Set-Location D:\Jaume\code\repo

$runs = @(
    @{ id = 'C1-competition';      extra = @('--slot-competition') },
    @{ id = 'C2-competition-iter3'; extra = @('--slot-competition', '--slot-iterations', '3') },
    @{ id = 'C3-comp-iter3-div';    extra = @('--slot-competition', '--slot-iterations', '3', '--slot-div-weight', '0.1') },
    @{ id = 'C4-comp-iter3-16slot'; extra = @('--slot-competition', '--slot-iterations', '3', '--num-slots', '16') },
    @{ id = 'C5-comp-attnpool';     extra = @('--slot-competition', '--slot-iterations', '3', '--slot-pool', 'attention') },
    @{ id = 'C6-comp-iter3-nogru';  extra = @('--slot-competition', '--slot-iterations', '3', '--no-slot-gru') }
)

$sw = [Diagnostics.Stopwatch]::StartNew()
foreach ($r in $runs) {
    $out = "D:\Jaume\output\mem\$($r.id)-s1"
    if (Test-Path "$out\final\config.json") {
        Write-Host "[fix] SKIP $($r.id) (exists)"
        continue
    }
    Write-Host "===================================================="
    Write-Host "[fix] START $($r.id)  elapsed $($sw.Elapsed.ToString('hh\:mm\:ss'))"
    $t = [Diagnostics.Stopwatch]::StartNew()
    & $py scripts/train_memory.py `
        --config configs/jaume-350m.json --pooling latent_memory --slot-heads 8 `
        --init-from D:\Jaume\350m-mlm --train-data D:\Jaume\data\all_nli.jsonl `
        --output $out --seed 1 --epochs 1 --batch-size 32 --grad-accum 4 `
        --lr 3e-4 --max-length 128 --log-every 200 --num-workers 0 @($r.extra) 2>&1 |
        Select-String -Pattern 'memory\]|Error|Traceback' | Select-Object -Last 6
    $t.Stop()
    Write-Host "[fix] DONE $($r.id) in $($t.Elapsed.ToString('mm\:ss'))  total $($sw.Elapsed.ToString('hh\:mm\:ss'))"
}
Write-Host "[fix] ALL DONE  total $($sw.Elapsed.ToString('hh\:mm\:ss'))"
