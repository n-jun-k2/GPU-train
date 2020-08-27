$(eval cuda := cuda10.1)
$(eval container := python-gpu)
curdir := $(shell pwd)

build:
	@docker build -t $(name):$(cuda) $(curdir)/Docker/$(name)/.

run:
	@docker run --gpus all --name $(container) -it $(name):$(cuda) /bin/bash

python-run:
	@docker run --gpus all --name $(container) -it -v $(curdir)/Docker/python/packages/:/tmp/python-app python:$(cuda) /bin/bash

test:
	@echo $(curdir)

clean:
	@docker stop $(shell docker ps -aq)
	@docker rm -f $(shell docker ps -aq)
	@docker rmi -f $(shell docker images -aq)