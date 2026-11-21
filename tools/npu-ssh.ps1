<#
.SYNOPSIS
  Run a command, or a script, on the Ascend NPU host over SSH.

.DESCRIPTION
  Everything host-specific — the tunnel address, the port, the user and the
  key path — is read from the environment, so none of it has to live in the
  repository:

    JAUME_NPU_HOST      ssh target, e.g. user@host      (required)
    JAUME_NPU_PORT      tcp port of the tunnel         (default 22)
    JAUME_NPU_KEY       path to the private key        (optional)
    JAUME_NPU_CANN      directory holding the CANN set_env.sh, e.g.
                         /opt/home/<user>/Ascend       (optional; when unset
                         the remote shell is used as-is)
    JAUME_NPU_WORKDIR   remote working directory       (default /tmp)

.EXAMPLE
  $env:JAUME_NPU_HOST = "user@127.0.0.1"
  $env:JAUME_NPU_PORT = "<tunnel port>"
  $env:JAUME_NPU_KEY  = "$env:USERPROFILE\.ssh\id_ed25519"
  powershell -File tools/npu-ssh.ps1 -ScriptFile prototype/actor/run_npu.sh -RunCmd "check"
#>
param(
  [Parameter(Mandatory=$false)][string]$Cmd,
  [Parameter(Mandatory=$false)][string]$ScriptFile,
  [Parameter(Mandatory=$false)][string]$RunCmd,
  [int]$TimeoutSec = 3600
)

$ErrorActionPreference = "Stop"

$host_ = $env:JAUME_NPU_HOST
if (-not $host_) { throw "set JAUME_NPU_HOST (e.g. user@127.0.0.1)" }
$port = if ($env:JAUME_NPU_PORT) { $env:JAUME_NPU_PORT } else { "22" }
$key  = $env:JAUME_NPU_KEY
$work = if ($env:JAUME_NPU_WORKDIR) { $env:JAUME_NPU_WORKDIR } else { "/tmp" }

$sshArgs = @("-p", $port,
             "-o", "StrictHostKeyChecking=no",
             "-o", "UserKnownHostsFile=NUL",
             "-o", "BatchMode=yes",
             "-o", "ConnectTimeout=20")
if ($key) { $sshArgs += @("-i", $key) }
$sshArgs += $host_

# Source the CANN environment when its location is known; otherwise the remote
# shell is used unchanged (useful for hosts that already export it).
$pre = ""
if ($env:JAUME_NPU_CANN) {
  $pre = "source $env:JAUME_NPU_CANN/ascend-toolkit/set_env.sh >/dev/null 2>&1; " +
         "source $env:JAUME_NPU_CANN/nnal/atb/set_env.sh >/dev/null 2>&1; "
}

if ($ScriptFile) {
  if (-not (Test-Path -LiteralPath $ScriptFile)) { throw "ScriptFile not found: $ScriptFile" }
  $remote = "$work/" + [System.IO.Path]::GetFileName($ScriptFile)
  $scpArgs = @("-P", $port, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL")
  if ($key) { $scpArgs += @("-i", $key) }
  $scpArgs += @($ScriptFile, "${host_}:$remote")
  & scp @scpArgs 2>&1 | Out-Null
  $cmd = $pre + "bash " + $remote + " " + $RunCmd
} else {
  if (-not $Cmd) { throw "give -Cmd or -ScriptFile" }
  $cmd = $pre + $Cmd
}

& ssh @sshArgs $cmd 2>&1
exit $LASTEXITCODE
