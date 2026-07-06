#!/usr/bin/env python3
"""抖音多账号 Chrome Profile 设置工具

用法:
  python3 login_setup.py --account fans --port 9224
  python3 login_setup.py --account sell --port 9225
  python3 login_setup.py --account shop --port 9223

第一次使用新账号时运行，扫码登录一次后永久有效。
"""
import argparse
import os
import subprocess
import sys
import time

PROFILES = {
    "fans": {
        "dir": os.path.expanduser("~/.douyin-fans-chrome"),
        "port": 9224,
        "desc": "吸粉号（纯内容，不挂链接）",
        "url": "https://creator.douyin.com",
    },
    "sell": {
        "dir": os.path.expanduser("~/.douyin-sell-chrome"),
        "port": 9225,
        "desc": "带货号（挂小黄车成交）",
        "url": "https://creator.douyin.com",
    },
    "shop": {
        "dir": os.path.expanduser("~/.jinritemai-chrome"),
        "port": 9223,
        "desc": "抖店运营（选品上架后台）",
        "url": "https://fxg.jinritemai.com",
    },
}

CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def is_cdp_ready(port: int) -> bool:
    try:
        r = subprocess.run(
            ["curl", "-s", "--max-time", "2", f"http://127.0.0.1:{port}/json/version"],
            capture_output=True, text=True, timeout=3,
        )
        return r.returncode == 0 and "Browser" in r.stdout
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="设置抖音多账号 Chrome Profile")
    parser.add_argument("--account", required=True, choices=list(PROFILES.keys()),
                        help="账号类型")
    parser.add_argument("--port", type=int, default=None,
                        help="CDP 端口（覆盖默认）")
    args = parser.parse_args()

    cfg = PROFILES[args.account]
    port = args.port or cfg["port"]
    profile_dir = cfg["dir"]

    print(f"\n{'='*50}")
    print(f"  账号: {args.account} — {cfg['desc']}")
    print(f"  Profile: {profile_dir}")
    print(f"  CDP Port: {port}")
    print(f"{'='*50}")
    print(f"\n首次设置流程：")
    print(f"  1. Chrome 将自动打开")
    print(f"  2. 扫码或手机号登录抖音创作者平台")
    print(f"  3. 登录成功后关闭 Chrome")
    print(f"  4. 以后此账号可全自动运行\n")

    # Launch Chrome with dedicated profile
    os.makedirs(profile_dir, exist_ok=True)
    cmd = [
        CHROME_BIN,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        cfg["url"],
    ]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait for CDP ready
    for i in range(30):
        if is_cdp_ready(port):
            print(f"  ✅ Chrome CDP ready on port {port}")
            break
        print(f"  等待 Chrome 启动... ({i+1}/30)")
        time.sleep(1)
    else:
        print("  ❌ Chrome 未能在 30s 内启动", file=sys.stderr)
        sys.exit(1)

    print(f"\n📍 请在 Chrome 中登录 {cfg['url']}")
    print(f"   登录完成后关闭 Chrome，登录态将保存在 {profile_dir}")
    print(f"   以后运行此账号无需再扫码。")

    # Wait for Chrome to close (user logs in and closes)
    input("\n按回车键确认已登录完成并关闭 Chrome...")

    # Verify profile saved
    cookie_files = []
    for root, dirs, files in os.walk(profile_dir):
        for f in files:
            if "Cookie" in f or "Token" in f or "Login" in f:
                cookie_files.append(os.path.join(root, f))

    if os.path.isdir(os.path.join(profile_dir, "Default")):
        print(f"\n  ✅ Profile 已保存: {profile_dir}")
        print(f"  账号 {args.account} 设置完成，下次运行将自动使用此登录态。")
    else:
        print(f"\n  ⚠️  Profile 目录存在但可能不完整")
        print(f"  如需重新设置，再次运行此命令")


if __name__ == "__main__":
    main()
