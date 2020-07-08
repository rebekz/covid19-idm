.PHONY: celan default virtualenv libs libs-test package build

VERSION_NUM := $(shell python __version__.py)
PACKAGE_NAME := idm-${VERSION_NUM}
IMAGE_NAME ?= eurekaai/ai_idm:v1

clean:
	rm -rf build/ dist/ test-results/ metastore_db/ spark-warehouse/
	find . -name '*.pyc' -exec rm {} \;
	find . -name '.pytest_cache' -o -name '__pycache__' -exec rm -r {} \; || true

# determine the current virtualenv
virtualenv:
ifeq ($(origin VIRTUAL_ENV), undefined)
    venv := ./.venv
else
    venv := $(VIRTUAL_ENV)
endif
#
# create virtualenv if not exists
${venv}:
	@virtualenv ${venv}

print-version:
	@echo ${VERSION_NUM}

build:
	mkdir -p build/archive/lib
	mkdir -p build/archive/temp
	python -m compileall fabric/
	cp -r fabric/ build/archive/temp/
	zip -i \*.pyc \*.py -r ./build/archive/lib/fabric.zip fabric/
	cp __version__.py build/archive/temp/
	poetry build -f wheel

package: build
	mkdir -p dist/${PACKAGE_NAME}/
	cp -r conf dist/${PACKAGE_NAME}/
	cp -r bin dist/${PACKAGE_NAME}/
	cp -r playbook dist/${PACKAGE_NAME}/
	cp manage.py dist/${PACKAGE_NAME}/
	cp dashboard.py dist/${PACKAGE_NAME}/
	cp README.md dist/${PACKAGE_NAME}/
	cp requirements.txt dist/${PACKAGE_NAME}/
	cp pyproject.toml dist/${PACKAGE_NAME}
	mkdir -p dist/${PACKAGE_NAME}/lib
	cp -r dist/*.whl dist/${PACKAGE_NAME}/lib
	mv ./build/archive/lib/* dist/${PACKAGE_NAME}/lib/
	make print-version > dist/${PACKAGE_NAME}/VERSION
	cd dist/ && tar czf ${PACKAGE_NAME}.tar.gz ${PACKAGE_NAME}
	rm -r dist/${PACKAGE_NAME}

libs: virtualenv ${venv}
	${venv}/bin/pip install -r requirements.txt -r requirements-test.txt

libs-test: virtualenv ${venv}
	${venv}/bin/pip install -r requirements-test.txt

test-nolibs: libs-test
	${venv}/bin/pytest -s tests/test_api.py tests/test_idm_factory.py tests/idm/models/ --junitxml=build/test-results/test-results_one.xml
	${venv}/bin/pytest -s tests/idm/test_utility.py --junitxml=build/test-results/test-results_two.xml

test: libs
	${venv}/bin/pytest -s tests/test_api.py tests/test_idm_factory.py tests/idm/models/ --junitxml=build/test-results/test-results_one.xml
	${venv}/bin/pytest -s tests/idm/test_utility.py --junitxml=build/test-results/test-results_two.xml


docker-test:
	docker run --rm -v ${PWD}:/code -ti -w /code \
		-e VIRTUAL_ENV=/miniconda \
		${IMAGE_NAME} make clean test-nolibs

docker-interactive:
	docker run --rm -v ${PWD}:/code -ti -w /code \
		-e VIRTUAL_ENV=/miniconda \
		${IMAGE_NAME}
