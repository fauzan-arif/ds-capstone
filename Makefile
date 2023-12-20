SHELL := /bin/zsh

VENV   = .venv
BIN    = $(VENV)/bin
PYTHON = $(VENV)/bin/python3
PIP    = $(VENV)/bin/pip3

PYTHON_VERSION := 3.11.7
PYENV_VERSION_INSTALLED := $(shell pyenv versions --bare | grep -e "^$(PYTHON_VERSION)$$")

log_success = (echo "\x1B[32m> $1\x1B[39m")
log_error   = (>&2 echo "\x1B[31m> $1\x1B[39m" && exit 1)
log_notice  = (echo "\x1b[33m> $1\x1B[39m")

.PHONY: setup

setup: clear_screen python venv pip requirements print_activation_message

python:	
	@echo "–– Installing Python –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"
	@if [ -z "$(PYENV_VERSION_INSTALLED)" ]; then \
		@echo "Installing Python $(PYTHON_VERSION) with pyenv..."; \
		pyenv install $(PYTHON_VERSION); \
	fi
	pyenv local $(PYTHON_VERSION)
	@$(call log_success, "Done!")

pip:
	@echo "–– Upgrading pip –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"
	@$(PIP) install --upgrade pip --quiet || $(call log_error, "ERROR: pip not upgraded!")
	@$(call log_success, "Done!")

requirements: requirements.txt
	@echo "–– Installing requirements –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"
	@$(PIP) install -r requirements.txt --quiet || $(call log_error, "ERROR: requirements not installed!")
	@$(call log_success, "Done!")

venv:
	@echo "–– Creating Virtual Environment ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"
	@test -d $(VENV) || python -m venv $(VENV)  || $(call log_error, "ERROR: $(VENV) not created!")
	@$(call log_success, "Done!")
	
clean:
	@echo "–– Removing Virtual Environment ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"
	@test $(VENV) && rm -rf .venv
	@$(call log_success, "Done!")
	
clear_screen:
	clear
	
print_activation_message:
	@echo
	@echo "–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"
	@echo "To activate your environment type:"
	@echo
	@$(call log_notice, "source .venv/bin/activate")
	@echo
	