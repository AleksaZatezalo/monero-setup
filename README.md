# xmr_miner_setup.py

One-shot bootstrap script that takes a fresh Debian/Ubuntu box and turns it into a tuned Monero CPU miner. Handles everything from kernel tuning to building XMRig from source, with optional P2Pool and local node support.

## Architecture

```
main()
├── preflight_checks()        — root, OS, architecture
├── detect_installed()        — probe for xmrig, monerod, p2pool
├── optimize_system()         — hugepages, MSR, CPU governor
├── install_dependencies()    — apt packages
├── install_monerod()         — official Monero CLI bundle
├── install_p2pool()          — download P2Pool release binary
├── install_xmrig()           — build from source (no donation hijack risk)
├── configure_xmrig()         — write config.json
├── start_monerod()           — local node (required for P2Pool)
├── start_p2pool()            — decentralised pool daemon
└── start_xmrig()             — begin mining
```

Each stage is idempotent — re-running the script skips components that are already installed.

## Requirements

- Debian 11+ / Ubuntu 20.04+ (Kali, Pop!\_OS also work)
- x86\_64 architecture
- Root privileges

## Quick Start

**Pool mining (fastest start):**

```bash
sudo python3 xmr_miner_setup.py --wallet <YOUR_WALLET_ADDRESS>
```

**P2Pool mining (decentralised, no pool fees):**

```bash
sudo python3 xmr_miner_setup.py --wallet <YOUR_WALLET_ADDRESS> --p2pool
```

> **Note:** P2Pool requires a synced local node. The script starts `monerod` automatically, but initial blockchain sync can take several hours. XMRig will connect and retry until P2Pool is ready.

## Usage

```
sudo python3 xmr_miner_setup.py --wallet <WALLET> [options]

Options:
  --wallet WALLET       Monero wallet address (required)
  --pool POOL           Mining pool (default: pool.hashvault.pro:443)
  --worker NAME         Worker/rig name (default: hostname)
  --threads N           CPU threads, 0 = all (default: 0)
  --donate-level N      XMRig dev donation % (default: 1)
  --no-hugepages        Skip hugepage setup
  --no-1gb              Skip 1GB hugepage GRUB config
  --run-node            Run a pruned monerod node
  --p2pool              Install and run P2Pool (implies --run-node)
  --p2pool-main         Use P2Pool main chain instead of mini
  --no-tls              Disable TLS to pool
  -v, --verbose         Debug logging
```

## Examples

```bash
# Pool mining with 4 threads, no dev donation
sudo python3 xmr_miner_setup.py --wallet 4XXXX... --threads 4 --donate-level 0

# P2Pool main chain (for high hashrate rigs)
sudo python3 xmr_miner_setup.py --wallet 4XXXX... --p2pool --p2pool-main

# Custom pool, no TLS
sudo python3 xmr_miner_setup.py --wallet 4XXXX... --pool xmr.pool.example:3333 --no-tls

# Local node only (no P2Pool)
sudo python3 xmr_miner_setup.py --wallet 4XXXX... --run-node
```

## What It Does to Your System

| Change | Reversible? | How to Undo |
|---|---|---|
| 2MB hugepages (2.5GB) | Yes | `echo 0 > /proc/sys/vm/nr_hugepages` |
| CPU governor → performance | Yes | `echo ondemand \| tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` |
| vm.swappiness → 10 | Yes | `sysctl vm.swappiness=60` |
| MSR kernel module loaded | Yes | `modprobe -r msr` |
| 1GB hugepages in GRUB | Reboot required | Remove `hugepagesz=1G hugepages=3` from `/etc/default/grub`, run `update-grub` |
| apt packages installed | Yes | Standard `apt remove` |

## Install Locations

| Component | Path |
|---|---|
| XMRig binary | `/opt/xmrig/xmrig` |
| XMRig config | `/etc/xmrig/config.json` |
| XMRig log | `/var/log/xmrig.log` |
| monerod binary | `/opt/monero/monerod` |
| monerod data | `/opt/monero/data/` |
| P2Pool binary | `/opt/p2pool/p2pool` |

## Managing Running Services

All processes run in detached `screen` sessions.

```bash
# Attach to a session
screen -r xmrig
screen -r p2pool
screen -r monerod

# Detach from session: Ctrl+A, then D

# Stop everything
screen -S xmrig -X quit
screen -S p2pool -X quit
screen -S monerod -X quit

# Check what's running
screen -ls
```

## P2Pool Notes

- `--p2pool` automatically implies `--run-node` since P2Pool needs a local monerod
- Default is the **mini** sidechain (lower share difficulty, better for < 50 KH/s)
- Use `--p2pool-main` only if you have significant hashrate
- XMRig config includes the remote pool as a fallback if P2Pool goes down
- P2Pool stratum listens on `127.0.0.1:3333`

## Security Notes

- XMRig is built from source at a pinned git tag to avoid pre-compiled binaries with tampered donation addresses
- When `--donate-level 0` is set, the script patches `donate.h` at compile time
- monerod runs with `--restricted-rpc` bound to localhost only
- All pool connections use TLS by default (disable with `--no-tls`)
