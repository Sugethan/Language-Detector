setup: requirements.txt
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

run:
	. venv/bin/activate && echo "./exemple_text.txt" | python3 LDetection.py

clean:
	rm -rf __pycache__

tests_run:
	echo "test_runs"

fclean: clean
	rm -rf venv
