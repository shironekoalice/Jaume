# JM-R: recurrent memory arms (plan section 8, stage 03).
#
# The v1 pooler had no gate option wired into a trained run, and B5/B6 were
# never trained. These are the cross-input arms:
#   B5-recurrent        : slot state carried across chunks, no gate
#   B6-recurrent-gate   : carried with an explicit gated update  (E5)
# and, given the round-2 finding that Slot Attention competition hurt on
# single-sentence tasks, a competition variant of each so the cross-input
# comparison is not confounded by the pooling normalisation:
#   B5r-recurrent-comp  : competition + 3 iterations, no gate
#   B6r-recurrent-gcomp : competition + 3 iterations + gate
#
# Same base/data/steps/lr/max_length as every other arm; seed 1.
$py = 'C:\Users\JingYue\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe'
$env:HF_ENDPOINT = 'https://hf-mirror.com'
Set-Location D:\Jaume\code\repo

$runs = @(
    @{ id = 'B5-recurrent';         extra = @('--pooling','recurrent_memory') },
    @{ id = 'B6-recurrent-gate';    extra = @('--pooling','recurrent_memory_gate') },
    @{ id = 'B5r-recurrent-comp';   extra = @('--pooling','recurrent_memory','--slot-competition','--slot-iterations','3') },
    @{ id = 'B6r-recurrent-gcomp';  extra = @('--pooling','recurrent_memory_gate','--slot-competition','--slot-iterations','3') }
)

foreach ($r in $runs) {
    $out = "D:\Jaume\output\mem\$($r.id)-s1"
    if (Test-Path "$out\final\config.json") { Write-Host "[jmr] SKIP $($r.id)"; continue }
    Write-Host "[jmr] START $($r.id) $(Get-Date -Format HH:mm:ss)"
    & $py scripts/train_memory.py --config configs/jaume-350m.json --slot-heads 8 `
        --num-slots 8 --init-from D:\Jaume\350m-mlm --train-data D:\Jaume\data\all_nli.jsonl `
        --output $out --seed 1 --epochs 1 --batch-size 32 --grad-accum 4 `
        --lr 3e-4 --max-length 128 --log-every 200 --num-workers 0 @($r.extra) 2>&1 |
        Select-String -Pattern 'memory\]|Error|Traceback' | Select-Object -Last 5 | ForEach-Object { $_.Line }
    Write-Host "[jmr] DONE $($r.id) $(Get-Date -Format HH:mm:ss)"
}
Write-Host "[jmr] ALL DONE $(Get-Date -Format HH:mm:ss)"
