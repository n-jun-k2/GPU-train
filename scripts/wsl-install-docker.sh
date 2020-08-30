#!/bin/sh
sudo apt -y update
sudo apt -y upgrade

curl https://get.docker.com | sh

sudo mkdir /sys/fs/cgroup/systemd
sudo mount -t cgroup -o none,name=systemd cgroup /sys/fs/cgroup/systemd

sudo service docker restart