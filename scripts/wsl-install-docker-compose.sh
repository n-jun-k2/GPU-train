#!/bin/sh

sudo curl -L "https://github.com/docker/compose/releases/download/v2.4.1/docker-compose-$(uname -s| tr '[:upper:]' '[:lower:]')-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose
