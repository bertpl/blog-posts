file_path=

help:
	@echo 'Commands:'
	@echo ''
	@echo '  format                         Format code using ruff (excluding notebooks).'
	@echo '  format-single-file             Format single file using ruff. Useful in e.g. pycharm to automatically trigger formatting on file save.'
	@echo ''
	@echo 'Options:'
	@echo ''
	@echo '  format-single-file             - accepts `file_path=<path>` to pass the relative path of the file to be formatted.'

format:
	ruff format .;
	ruff check --fix .;

format-single-file:
	ruff format ${file_path};
	ruff check --fix ${file_path};