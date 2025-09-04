# 🐧 Linux + 🐍 Python Cheat Sheet

---

## 🔹 Linux Basics

### File & Directory Navigation
- `pwd` → show current directory  
- `ls` → list files  
- `ls -l` → detailed list  
- `ls -a` → show hidden files  
- `cd foldername/` → change directory  
- `cd ..` → go back one directory  
- `cd ~` → go to home directory  

### File Management
- `cp file1 file2` → copy file  
- `mv file1 file2` → move/rename file  
- `rm file.txt` → delete file  
- `rm -r folder/` → delete folder  
- `touch file.txt` → create new file  
- `mkdir folder` → create new folder  

### Viewing Files
- `cat file.txt` → show file contents  
- `less file.txt` → view file page by page  
- `nano file.txt` → edit file in terminal  

### System & Packages
- `sudo apt update` → update package list  
- `sudo apt upgrade` → upgrade packages  
- `sudo apt install pkgname` → install package  
- `sudo apt remove pkgname` → remove package  

---

## 🔹 Python Basics

### Running Python
- `python3 script.py` → run script  
- `python3` → start interactive shell  
- `exit()` → quit shell  

### Virtual Environment
```bash
python3 -m venv myenv        # create venv
source myenv/bin/activate    # activate
deactivate                   # exit
