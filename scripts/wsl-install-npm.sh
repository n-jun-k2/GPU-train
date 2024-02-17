#!/bin/sh

sudo apt update
sudo apt install nodejs npm
sudo npm -g install n
sudo n stable
sudo apt purge nodejs npm
sudo apt autoremove