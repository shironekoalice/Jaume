# Jaume-Memory stage-01 sweep: core JM-S arms x 3 seeds.
# Fairness: same base (350m-mlm), data, optimiser, steps, max_length; seed varies.
$py = 'C:\Users\JingYue\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe'
$env:HF_ENDPOINT = 'https://hf-mirror.com'
Set-Location D:\Jaume\code\repo

$arms = @(
    @{ id = 'B0-mean';        pooling = 'mean' },
    @{ id = 'B1-cls';         pooling = 'cls' },
    @{ id = 'B2-attention';   pooling = 'attention' },
    @{ id = 'B2mlp-attnmlp';  pooling = 'attention_mlp' },
    @{ id = 'B3-latent';      pooling = 'latent_memory' }
)
$seeds = @(1, 2, 3)

$sw = [Diagnostics.Stopwatch]::StartNew()
foreach ($arm in $arms) {
    foreach ($seed in $seeds) {
        $out = "D:\Jaume\output\mem\$($arm.id)-s$seed"
        if (Test-Path "$out\final\config.json") {
            Write-Host "[sweep] SKIP $($arm.id) seed $seed (exists)"
            continue
        }
        Write-Host "===================================================="
        Write-Host "[sweep] START $($arm.id) seed $seed  elapsed $($sw.Elapsed.ToString('hh\:mm\:ss'))"
        $t = [Diagnostics.Stopwatch]::StartNew()
        & $py scripts/train_memory.py `
            --config configs/jaume-350m.json `
            --pooling $arm.pooling --num-slots 8 --slot-heads 8 `
            --init-from D:\Jaume\350m-mlm `
            --train-data D:\Jaume\data\all_nli.jsonl `
            --output $out --seed $seed `
            --epochs 1 --batch-size 32 --grad-accum 4 --lr 3e-4 `
            --max-length 128 --log-every 100 --num-workers 0 2>&1 |
            Select-String -Pattern 'memory\]|Error|Traceback' | Select-Object -Last 8
        $t.Stop()
        Write-Host "[sweep] DONE $($arm.id) seed $seed in $($t.Elapsed.ToString('mm\:ss'))  total $($sw.Elapsed.ToString('hh\:mm\:ss'))"
    }
}
Write-Host "[sweep] ALL TRAINING DONE  total $($sw.Elapsed.ToString('hh\:mm\:ss'))"
