# Setup Pi
**Objective:** Prepare Raspberry Pi for development (without monitor).

## Prerequisites
- SD card reader
- Raspberry Pi Imager [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
- USB mic

## Install Raspberry Pi imager  
<img width="628" height="445" alt="image" src="https://github.com/user-attachments/assets/422372f5-342e-452b-91db-6b5eb7da16a7" />  

## Enable SSH  
<img width="592" height="395" alt="image" src="https://github.com/user-attachments/assets/e31fc60d-917f-420c-a053-76cccd380c18" />  

## Create username and password and connected network
Username and password are used to login Raspberry Pi (like login into your Window).
Network used here is to create a network connection for VNC and Internet. Hotspot is used in my case so the IP of Raspberry Pi easy to obtain.

<img width="599" height="394" alt="image" src="https://github.com/user-attachments/assets/99dd730f-e58b-4569-a9d2-b7a90ccd85af" />  

## Erase SD to write in Raspberry Pi OS
<img width="598" height="399" alt="image" src="https://github.com/user-attachments/assets/d413f394-21fa-44b8-b0df-3e16e260cc7d" />  

## Finish installing Raspberry Pi OS
<img width="604" height="395" alt="image" src="https://github.com/user-attachments/assets/e17f1353-b4e6-4dda-958a-1c77aed386ed" />  

## Obtain IP of Raspberry Pi from Phone
To obtain IP of Raspberry Pi, check phone's Hotspot settings where can monitor the devices connected then look for details of the devices named raspbeerypi

# Configure Raspberry Pi
## log into Raspberrt Pi via command prompt (cmd)
Use username and password that is created previously.
**Password will not show up, just type and enter**
<img width="1588" height="847" alt="image" src="https://github.com/user-attachments/assets/1d7a3cbb-5894-46d7-b4ed-afecb1cdeba2" />

## raspi-config
open config tool
```bash
sudo raspi-config
```
<img width="732" height="391" alt="image" src="https://github.com/user-attachments/assets/6b55a440-2375-4df9-aaed-4b5ff008e48e" />

## Enable VNC
