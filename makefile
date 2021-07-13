$(eval cuda := cuda10.1)
curdir := $(shell pwd)

build:
	@docker build -t $(name):$(cuda) $(curdir)/Docker/$(name)/.

python-build:
	@make build name=python

cpp-build:
	@make build name=cpp

vulkan-build:
	@make build name=vulkan

run:
	@docker run --gpus all --name $(container)-gpu -it --rm -v $(mount):/tmp $(name):$(cuda) /bin/bash

python-run:
	@make run container=python mount=$(curdir)/Docker/python/packages/ name=python

cpp-run:
	@make run container=cpp mount=$(curdir)/Docker/cpp/src name=cpp

vulkan-run:
	@docker run --gpus all --name vulkan-gpu  -it --rm -v $(curdir)/Docker/vulkan/src:/tmp/project -v $(curdir)/Docker/vulkan/template:/tmp/template vulkan:$(cuda) /bin/bash

python-all: python-build python-run

cpp-all: cpp-build cpp-run

vulkan-all: vulkan-build vulkan-run

test:
	@echo $(curdir)

clean:
	@docker rmi -f $(shell docker images -aq)
	@docker volume rm $(shell docker volume ls -q)
	@docker network prune