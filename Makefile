REMOTE_MACHINE=3.123.206.99
PORT=50051

dev-env:
	conda env update -f dev-environment.yaml

.PHONY: proto
proto: 
	@# https://github.com/grpc/grpc/issues/9575#issuecomment-293934506
	python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/src/service/*.proto

docker-server: 
	DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 docker-compose up --build freeform

local-client:
	python -m src.service.inpainting_client $(REMOTE_MACHINE) $(PORT)

docker-client:
	DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 docker-compose build freeform && \
		docker-compose run --entrypoint "python -m service.inpainting_client freeform 50051" freeform