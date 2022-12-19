lint:
	black .
	flake8 .

push:
	make lint
	git add .
	git commit
	git push origin HEAD