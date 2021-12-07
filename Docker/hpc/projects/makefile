.ONESHELL:
.PHONY: clean build project

clean/%:
	-rm -r ./$(@F)/build

build/%:
	cmake -S ./$(@F) -B ./$(@F)/build
	cmake --build ./$(@F)/build

project/%:
	cp -r /tmp/templates/cuda_project $(@F)
	sed -i -e "2i project($(@F))" $(@F)/CMakeLists.txt