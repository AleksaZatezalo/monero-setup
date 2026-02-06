#!/usr/bin/env python3
"""
xmr_miner_setup.py — Debian Monero Mining & Node Bootstrap

Top-down architecture:
    main()
    ├── preflight_checks()        — root, OS, architecture
    ├── detect_installed()        — probe for xmrig, monerod, p2pool
    ├── optimize_system()         — hugepages, MSR, CPU governor
    ├── install_dependencies()    — apt packages
    ├── install_monerod()         — official Monero CLI bundle
    ├── install_p2pool()          — download P2Pool release binary
    ├── install_xmrig()          — build from source (no donation hijack risk)
    ├── configure_xmrig()        — write config.json
    ├── start_monerod()          — local node (required for P2Pool)
    ├── start_p2pool()           — decentralised pool daemon
    └── start_xmrig()           — begin mining

Usage:
    sudo python3 xmr_miner_setup.py --wallet <YOUR_WALLET_ADDRESS> [options]

Requires: Debian 11+ / Ubuntu 20.04+, root privileges.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

XMRIG_GIT      = "https://github.com/xmrig/xmrig.git"
XMRIG_VERSION  = "v6.22.2"
MONERO_VERSION  = "0.18.3.4"
MONERO_URL_TPL  = (
    "https://downloads.getmonero.org/cli/"
    "monero-linux-x64-v{version}.tar.bz2"
)

DEFAULT_POOL    = "pool.hashvault.pro:443"
DEFAULT_COIN    = "monero"
INSTALL_PREFIX  = Path("/opt/monero")
XMRIG_PREFIX    = Path("/opt/xmrig")
P2POOL_PREFIX   = Path("/opt/p2pool")
CONFIG_DIR      = Path("/etc/xmrig")

P2POOL_VERSION  = "v4.2"
P2POOL_URL_TPL  = (
    "https://github.com/SChernykh/p2pool/releases/download/"
    "{version}/p2pool-{version}-linux-x64.tar.gz"
)
P2POOL_STRATUM  = "127.0.0.1:3333"

LOG = logging.getLogger("xmr-setup")

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class MinerConfig:
    """All tunables in one place."""
    wallet: str
    pool: str              = DEFAULT_POOL
    worker_name: str       = platform.node() or "debian-miner"
    threads: int           = 0  # 0 = auto
    coin: str              = DEFAULT_COIN
    tls: bool              = True
    donate_level: int      = 1
    hugepages: bool        = True
    one_gb_pages: bool     = True
    run_node: bool         = False
    use_p2pool: bool       = False
    p2pool_mini: bool      = True
    p2pool_prefix: Path    = P2POOL_PREFIX
    node_rpc: str          = "127.0.0.1:18081"
    install_prefix: Path   = INSTALL_PREFIX
    xmrig_prefix: Path     = XMRIG_PREFIX

@dataclass
class SystemState:
    """Detected state of the host."""
    is_root: bool              = False
    is_debian: bool            = False
    arch: str                  = ""
    xmrig_path: Optional[str]  = None
    monerod_path: Optional[str] = None
    p2pool_path: Optional[str]  = None
    cpu_threads: int           = os.cpu_count() or 1
    hugepages_enabled: bool    = False
    one_gb_supported: bool     = False
    msr_available: bool        = False

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def run(cmd: str | list[str], check: bool = True, capture: bool = False, **kw) -> subprocess.CompletedProcess:
    """Thin wrapper around subprocess.run with logging."""
    if isinstance(cmd, str):
        cmd_list = cmd.split()
    else:
        cmd_list = cmd
    LOG.debug("$ %s", " ".join(cmd_list))
    return subprocess.run(
        cmd_list,
        check=check,
        capture_output=capture,
        text=True,
        **kw,
    )

def cmd_exists(name: str) -> Optional[str]:
    return shutil.which(name)

def read_file(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except (OSError, PermissionError):
        return ""

def write_sysctl(key: str, value: str) -> None:
    path = f"/proc/sys/{key.replace('.', '/')}"
    try:
        Path(path).write_text(value)
        LOG.info("sysctl %s = %s", key, value)
    except OSError as exc:
        LOG.warning("Failed to set %s: %s", key, exc)

# ---------------------------------------------------------------------------
# 1. Preflight
# ---------------------------------------------------------------------------

def preflight_checks(state: SystemState) -> None:
    """Validate we're on a supported Debian system with root."""
    state.is_root = os.geteuid() == 0
    if not state.is_root:
        LOG.error("This script must be run as root (sudo).")
        sys.exit(1)

    state.arch = platform.machine()
    if state.arch not in ("x86_64", "amd64"):
        LOG.error("Only x86_64 is supported. Detected: %s", state.arch)
        sys.exit(1)

    os_release = read_file("/etc/os-release").lower()
    state.is_debian = any(d in os_release for d in ("debian", "ubuntu", "kali", "pop"))
    if not state.is_debian:
        LOG.warning("Non-Debian OS detected — proceeding anyway, YMMV.")

    LOG.info("Preflight OK — arch=%s, cpus=%d", state.arch, state.cpu_threads)

# ---------------------------------------------------------------------------
# 2. Detection
# ---------------------------------------------------------------------------

def detect_installed(state: SystemState) -> None:
    """Check what's already on the system."""
    state.xmrig_path = (
        cmd_exists("xmrig")
        or str(XMRIG_PREFIX / "xmrig") if (XMRIG_PREFIX / "xmrig").exists() else None
    )
    state.monerod_path = (
        cmd_exists("monerod")
        or str(INSTALL_PREFIX / "monerod") if (INSTALL_PREFIX / "monerod").exists() else None
    )
    state.p2pool_path = (
        cmd_exists("p2pool")
        or str(P2POOL_PREFIX / "p2pool") if (P2POOL_PREFIX / "p2pool").exists() else None
    )

    # Hugepages
    hp = read_file("/sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages")
    state.hugepages_enabled = hp.isdigit() and int(hp) > 0

    # 1GB page support (cpu flag: pdpe1gb)
    cpuinfo = read_file("/proc/cpuinfo")
    state.one_gb_supported = "pdpe1gb" in cpuinfo

    # MSR module
    state.msr_available = Path("/dev/cpu/0/msr").exists() or cmd_exists("modprobe") is not None

    LOG.info(
        "Detected — xmrig=%s  monerod=%s  p2pool=%s  hugepages=%s  1gb=%s  msr=%s",
        bool(state.xmrig_path),
        bool(state.monerod_path),
        bool(state.p2pool_path),
        state.hugepages_enabled,
        state.one_gb_supported,
        state.msr_available,
    )

# ---------------------------------------------------------------------------
# 3. System optimizations
# ---------------------------------------------------------------------------

def optimize_system(state: SystemState, cfg: MinerConfig) -> None:
    """Apply kernel and CPU tuning for RandomX."""
    LOG.info("=== Applying system optimizations ===")

    # --- 2MB Hugepages (required by RandomX) ---
    if cfg.hugepages:
        # 1280 pages × 2MB = 2.5GB — enough for RandomX dataset + some headroom
        desired_pages = 1280
        write_sysctl("vm.nr_hugepages", str(desired_pages))
        LOG.info("Allocated %d × 2MB hugepages", desired_pages)

    # --- 1GB Hugepages (major perf boost) ---
    if cfg.one_gb_pages and state.one_gb_supported:
        LOG.info("1GB hugepage support detected — configuring GRUB")
        _configure_grub_hugepages()
    elif cfg.one_gb_pages:
        LOG.warning("CPU does not support 1GB pages (pdpe1gb flag missing), skipping.")

    # --- MSR (wrmsr for RandomX boost) ---
    _enable_msr(state)

    # --- CPU performance governor ---
    _set_cpu_governor("performance")

    # --- vm.swappiness (minimise swap interference) ---
    write_sysctl("vm.swappiness", "10")

    LOG.info("=== Optimizations applied ===")


def _configure_grub_hugepages() -> None:
    """Append hugepagesz=1G to GRUB_CMDLINE_LINUX_DEFAULT if not present."""
    grub_file = Path("/etc/default/grub")
    if not grub_file.exists():
        LOG.warning("/etc/default/grub not found, skipping 1GB pages GRUB config.")
        return

    content = grub_file.read_text()
    flag = "hugepagesz=1G hugepages=3"  # 3 × 1GB
    if flag in content:
        LOG.info("1GB hugepages already in GRUB config.")
        return

    # Append inside GRUB_CMDLINE_LINUX_DEFAULT
    import re
    new_content, count = re.subn(
        r'(GRUB_CMDLINE_LINUX_DEFAULT="[^"]*)',
        rf"\1 {flag}",
        content,
        count=1,
    )
    if count:
        grub_file.write_text(new_content)
        run("update-grub", check=False)
        LOG.info("Added 1GB hugepages to GRUB. Reboot required to take effect.")
    else:
        LOG.warning("Could not patch GRUB config automatically.")


def _enable_msr(state: SystemState) -> None:
    """Load msr module and apply RandomX MSR tweaks."""
    if not state.msr_available:
        return
    run("modprobe msr", check=False)
    # XMRig handles MSR writes at startup with --randomx-wrmsr, but we
    # ensure the module is loaded so it can.
    LOG.info("MSR kernel module loaded.")


def _set_cpu_governor(governor: str) -> None:
    """Set all CPU cores to the given frequency governor."""
    gov_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if not gov_path.exists():
        LOG.warning("cpufreq governor interface not found (VM or no driver).")
        return

    for p in Path("/sys/devices/system/cpu/").glob("cpu[0-9]*/cpufreq/scaling_governor"):
        try:
            p.write_text(governor)
        except OSError:
            pass
    LOG.info("CPU governor set to '%s'.", governor)

# ---------------------------------------------------------------------------
# 4. Install dependencies
# ---------------------------------------------------------------------------

APT_PACKAGES = [
    "git", "build-essential", "cmake", "automake", "libtool", "autoconf",
    "libssl-dev", "libhwloc-dev", "libuv1-dev",
    "wget", "tar", "bzip2", "screen",
]

def install_dependencies() -> None:
    """Ensure build tools and libraries are present."""
    LOG.info("Installing apt dependencies …")
    run("apt-get update -qq")
    run(["apt-get", "install", "-y", "-qq"] + APT_PACKAGES)
    LOG.info("Dependencies installed.")

# ---------------------------------------------------------------------------
# 5. Install monerod (optional local node)
# ---------------------------------------------------------------------------

def install_monerod(cfg: MinerConfig) -> str:
    """Download the official Monero CLI tarball and extract monerod."""
    dest = cfg.install_prefix
    monerod_bin = dest / "monerod"
    if monerod_bin.exists():
        LOG.info("monerod already present at %s", monerod_bin)
        return str(monerod_bin)

    dest.mkdir(parents=True, exist_ok=True)
    url = MONERO_URL_TPL.format(version=MONERO_VERSION)
    archive = dest / "monero.tar.bz2"

    LOG.info("Downloading Monero CLI from %s …", url)
    urllib.request.urlretrieve(url, str(archive))

    run(f"tar -xjf {archive} -C {dest} --strip-components=1")
    archive.unlink()
    LOG.info("monerod installed → %s", monerod_bin)
    return str(monerod_bin)

# ---------------------------------------------------------------------------
# 6. Install XMRig (build from source for trust & latest RandomX)
# ---------------------------------------------------------------------------

def install_xmrig(cfg: MinerConfig) -> str:
    """Clone and compile XMRig from source."""
    xmrig_bin = cfg.xmrig_prefix / "xmrig"
    if xmrig_bin.exists():
        LOG.info("xmrig already present at %s", xmrig_bin)
        return str(xmrig_bin)

    src = Path("/tmp/xmrig-src")
    if src.exists():
        shutil.rmtree(src)

    LOG.info("Cloning XMRig %s …", XMRIG_VERSION)
    run(["git", "clone", "--depth", "1", "--branch", XMRIG_VERSION, XMRIG_GIT, str(src)])

    # Optional: zero out the hard-coded dev donation to put you in full control.
    # The default 1% is fair — change donate_level in config if you want to keep it.
    donate_h = src / "src" / "donate.h"
    if donate_h.exists() and cfg.donate_level == 0:
        original = donate_h.read_text()
        patched = original.replace(
            "constexpr const int kDefaultDonateLevel = 1;",
            "constexpr const int kDefaultDonateLevel = 0;",
        ).replace(
            "constexpr const int kMinimumDonateLevel = 1;",
            "constexpr const int kMinimumDonateLevel = 0;",
        )
        donate_h.write_text(patched)
        LOG.info("Patched donate.h (donate_level=0).")

    build = src / "build"
    build.mkdir()
    LOG.info("Building XMRig (cmake + make) …")
    run(f"cmake .. -DWITH_HWLOC=ON -DWITH_TLS=ON", cwd=build)
    run(f"make -j{os.cpu_count()}", cwd=build)

    cfg.xmrig_prefix.mkdir(parents=True, exist_ok=True)
    shutil.copy2(build / "xmrig", xmrig_bin)
    xmrig_bin.chmod(0o755)

    # Cleanup
    shutil.rmtree(src, ignore_errors=True)

    LOG.info("xmrig built → %s", xmrig_bin)
    return str(xmrig_bin)

# ---------------------------------------------------------------------------
# 6b. Install P2Pool (download release binary)
# ---------------------------------------------------------------------------

def install_p2pool(cfg: MinerConfig) -> str:
    """Download and extract P2Pool release binary."""
    p2pool_bin = cfg.p2pool_prefix / "p2pool"
    if p2pool_bin.exists():
        LOG.info("p2pool already present at %s", p2pool_bin)
        return str(p2pool_bin)

    cfg.p2pool_prefix.mkdir(parents=True, exist_ok=True)
    url = P2POOL_URL_TPL.format(version=P2POOL_VERSION)
    archive = cfg.p2pool_prefix / "p2pool.tar.gz"

    LOG.info("Downloading P2Pool %s from %s …", P2POOL_VERSION, url)
    urllib.request.urlretrieve(url, str(archive))

    run(f"tar -xzf {archive} -C {cfg.p2pool_prefix} --strip-components=1")
    archive.unlink()

    if not p2pool_bin.exists():
        # Some releases nest the binary one level deeper
        for candidate in cfg.p2pool_prefix.rglob("p2pool"):
            if candidate.is_file():
                shutil.move(str(candidate), str(p2pool_bin))
                break

    p2pool_bin.chmod(0o755)
    LOG.info("p2pool installed → %s", p2pool_bin)
    return str(p2pool_bin)


def start_p2pool(cfg: MinerConfig, p2pool_bin: str) -> None:
    """Launch P2Pool in a detached screen session."""
    cmd = [
        "screen", "-dmS", "p2pool",
        p2pool_bin,
        "--host", "127.0.0.1",
        "--rpc-port", "18081",
        "--wallet", cfg.wallet,
        "--stratum", P2POOL_STRATUM,
    ]
    if cfg.p2pool_mini:
        cmd.append("--mini")

    run(cmd)
    LOG.info("p2pool started in screen session 'p2pool'.")
    LOG.info("  Sidechain: %s", "mini" if cfg.p2pool_mini else "main")
    LOG.info("  Stratum:   %s", P2POOL_STRATUM)
    LOG.info("  Attach:    screen -r p2pool")

# ---------------------------------------------------------------------------
# 7. Configure XMRig
# ---------------------------------------------------------------------------

def configure_xmrig(cfg: MinerConfig) -> Path:
    """Write /etc/xmrig/config.json."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = CONFIG_DIR / "config.json"

    pool_entry = {
        "url": cfg.pool,
        "user": cfg.wallet,
        "pass": cfg.worker_name,
        "rig-id": cfg.worker_name,
        "tls": cfg.tls,
        "keepalive": True,
        "nicehash": False,
    }

    # If running a local node, add it as a higher-priority (solo) pool
    pools = []
    if cfg.use_p2pool:
        # P2Pool exposes a stratum server on localhost — no TLS, no daemon mode
        pools.append({
            "url": P2POOL_STRATUM,
            "user": cfg.wallet,
            "pass": cfg.worker_name,
            "rig-id": cfg.worker_name,
            "tls": False,
            "keepalive": True,
            "nicehash": False,
        })
        # Keep the remote pool as fallback in case P2Pool goes down
        pools.append(pool_entry)
    elif cfg.run_node:
        pools.append({
            "url": f"127.0.0.1:18081",
            "user": cfg.wallet,
            "pass": cfg.worker_name,
            "daemon": True,       # daemon (solo) mode
            "tls": False,
        })
    pools.append(pool_entry)

    config = {
        "autosave": True,
        "cpu": {
            "enabled": True,
            "huge-pages": True,
            "huge-pages-jit": True,
            "hw-aes": None,      # auto-detect
            "priority": None,
            "yield": True,
            "max-threads-hint": 100 if cfg.threads == 0 else None,
        },
        "randomx": {
            "init": -1,
            "init-avx2": -1,
            "mode": "auto",
            "1gb-pages": cfg.one_gb_pages,
            "rdmsr": True,
            "wrmsr": True,
            "numa": True,
        },
        "donate-level": cfg.donate_level,
        "pools": pools,
        "log-file": "/var/log/xmrig.log",
        "print-time": 60,
    }

    config_path.write_text(json.dumps(config, indent=2))
    LOG.info("Config written → %s", config_path)
    return config_path

# ---------------------------------------------------------------------------
# 8. Start services
# ---------------------------------------------------------------------------

def start_monerod(cfg: MinerConfig, monerod_bin: str) -> None:
    """Launch monerod in a detached screen session."""
    data_dir = cfg.install_prefix / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "screen", "-dmS", "monerod",
        monerod_bin,
        "--data-dir", str(data_dir),
        "--rpc-bind-ip", "127.0.0.1",
        "--rpc-bind-port", "18081",
        "--confirm-external-bind",
        "--non-interactive",
        "--restricted-rpc",
        "--prune-blockchain",       # saves ~2/3 disk
        "--sync-pruned-blocks",
        "--db-sync-mode", "safe",
    ]
    run(cmd)
    LOG.info("monerod started in screen session 'monerod'.")
    LOG.info("  Attach: screen -r monerod")


def start_xmrig(xmrig_bin: str, config_path: Path) -> None:
    """Launch xmrig in a detached screen session."""
    cmd = [
        "screen", "-dmS", "xmrig",
        xmrig_bin,
        "--config", str(config_path),
    ]
    run(cmd)
    LOG.info("xmrig started in screen session 'xmrig'.")
    LOG.info("  Attach: screen -r xmrig")

# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bootstrap a Debian machine for Monero mining.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  sudo python3 xmr_miner_setup.py --wallet 4XXXXXXX… --pool pool.hashvault.pro:443",
    )
    p.add_argument("--wallet", required=True, help="Your Monero wallet address")
    p.add_argument("--pool", default=DEFAULT_POOL, help=f"Mining pool (default: {DEFAULT_POOL})")
    p.add_argument("--worker", default=platform.node(), help="Worker/rig name")
    p.add_argument("--threads", type=int, default=0, help="CPU threads (0 = all)")
    p.add_argument("--donate-level", type=int, default=1, help="XMRig dev donation %% (0–100)")
    p.add_argument("--no-hugepages", action="store_true", help="Skip hugepage setup")
    p.add_argument("--no-1gb", action="store_true", help="Skip 1GB hugepage GRUB config")
    p.add_argument("--run-node", action="store_true", help="Also run a pruned monerod node")
    p.add_argument("--p2pool", action="store_true", help="Install and run P2Pool (implies --run-node)")
    p.add_argument("--p2pool-main", action="store_true", help="Use P2Pool main chain instead of mini")
    p.add_argument("--no-tls", action="store_true", help="Disable TLS to pool")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = MinerConfig(
        wallet=args.wallet,
        pool=args.pool,
        worker_name=args.worker,
        threads=args.threads,
        donate_level=args.donate_level,
        hugepages=not args.no_hugepages,
        one_gb_pages=not args.no_1gb,
        tls=not args.no_tls,
        run_node=args.run_node or args.p2pool,  # P2Pool requires a local node
        use_p2pool=args.p2pool,
        p2pool_mini=not args.p2pool_main,
    )

    state = SystemState()

    # ── Pipeline ──────────────────────────────────────────────
    preflight_checks(state)
    detect_installed(state)
    optimize_system(state, cfg)
    install_dependencies()

    # Monerod (optional, or required by P2Pool)
    monerod_bin = None
    if cfg.run_node and not state.monerod_path:
        monerod_bin = install_monerod(cfg)
    elif state.monerod_path:
        monerod_bin = state.monerod_path

    # P2Pool (optional)
    p2pool_bin = None
    if cfg.use_p2pool:
        p2pool_bin = state.p2pool_path or install_p2pool(cfg)

    # XMRig
    xmrig_bin = state.xmrig_path or install_xmrig(cfg)

    # Configure & launch
    config_path = configure_xmrig(cfg)

    if cfg.run_node and monerod_bin:
        start_monerod(cfg, monerod_bin)

    if cfg.use_p2pool and p2pool_bin:
        start_p2pool(cfg, p2pool_bin)

    start_xmrig(xmrig_bin, config_path)

    LOG.info("════════════════════════════════════════")
    LOG.info("  Mining started! Pool: %s", cfg.pool if not cfg.use_p2pool else f"P2Pool ({'mini' if cfg.p2pool_mini else 'main'})")
    LOG.info("  Wallet: %s…%s", cfg.wallet[:8], cfg.wallet[-8:])
    LOG.info("  Attach to miner:  screen -r xmrig")
    if cfg.use_p2pool:
        LOG.info("  Attach to p2pool: screen -r p2pool")
    LOG.info("  Logs:             tail -f /var/log/xmrig.log")
    LOG.info("════════════════════════════════════════")


if __name__ == "__main__":
    main()