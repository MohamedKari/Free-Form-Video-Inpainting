docker-run:
	DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 docker-compose build freeform
	DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 docker-compose run freeform
	echo "Run 'docker-compose run singleshotpose' to spin up an interactive session!"