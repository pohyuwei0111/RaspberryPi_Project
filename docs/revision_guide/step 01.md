# STEP 01 — Setup Pi (headless + VNC)
**Objective:** Prepare Raspberry Pi for development (headless).

## Prerequisites
- SD card with Raspberry Pi OS
- Network access
- USB mic

## Commands (copy/paste)
```bash
# flash and boot (handled separately)
# enable SSH & VNC from `raspi-config` or via /boot
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git alsa-utils sox
arecord -l           # list recording devices
