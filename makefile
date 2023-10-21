$(eval cuda := cuda11.6)
curdir := $(shell pwd)

include ./Docker/q/makefile

service:
	@docker compose up -d $(servicename) --build

clean:
	@docker builder prune
	@docker rmi -f $(shell docker images -aq)
	@docker volume rm $(shell docker volume ls -q)
	@docker network prune