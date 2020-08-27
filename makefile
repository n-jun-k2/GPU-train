$(eval cuda := cuda10.1)
curdir := $(shell pwd)

build:
	@docker build -t $(name):$(cuda) $(curdir)/Docker/$(name)/.

run:
	@docker run --gpus all --name $(container)-gpu -it $(name):$(cuda) /bin/bash

python-run:
	@docker run --gpus all --name python-gpu -it -v $(curdir)/Docker/python/packages/:/tmp/python-app python:$(cuda) /bin/bash

c-run:
	@docker run --gpus all --name c-gpu -it -v $(curdir)/Docker/c/src/:/tmp/src c:$(cuda) /bin/bash

test:
	@echo $(curdir)

clean:
	@docker stop $(shell docker ps -aq)
	@docker rm -f $(shell docker ps -aq)
	@docker rmi -f $(shell docker images -aq)
	@docker volume rm $(shell docker volume ls -q)