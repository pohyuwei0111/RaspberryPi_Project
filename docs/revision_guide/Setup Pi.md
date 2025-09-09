# Setup Pi (headless + VNC)
**Objective:** Prepare Raspberry Pi for development (without monitor).

## Prerequisites
- SD card reader
- Raspberry Pi Imager [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
- USB mic

<img width="628" height="445" alt="image" src="https://github.com/user-attachments/assets/422372f5-342e-452b-91db-6b5eb7da16a7" />


## Commands (copy/paste)
```bash
# flash and boot (handled separately)
# enable SSH & VNC from `raspi-config` or via /boot
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git alsa-utils sox
arecord -l           # list recording devices
