#!/usr/bin/bash
# run_tests.sh

COVERAGE_THRESHOLD=90

TERM=${TERM:-xterm}

# set up terminal colors
NORMAL=$(tput sgr0)
RED=$(tput bold && tput setaf 1)
GREEN=$(tput bold && tput setaf 2)
YELLOW=$(tput bold && tput setaf 3)

check_python_version() {
    python3 tools/check_python_version.py 3 6
}

function prepare_venv() {
	# we want tests to run on python3.6
	printf 'checking alias `python3.6` ... ' >&2
	PYTHON=$(which python3.6 2> /dev/null)
	if [ "$?" -ne "0" ]; then
		printf "%sNOT FOUND%s\n" "${YELLOW}" "${NORMAL}" >&2

		printf 'checking alias `python3` ... ' >&2
		PYTHON=$(which python3 2> /dev/null)

		let ec=$?
		[ "$ec" -ne "0" ] && printf "${RED} NOT FOUND ${NORMAL}\n" && return $ec
	fi

	printf "%sOK%s\n" "${GREEN}" "${NORMAL}" >&2

	${PYTHON} -m venv "venv" && source venv/bin/activate
}

check_python_version

[ "$NOVENV" == "1" ] || prepare_venv || exit 1

# install nvdlib
git clone https://github.com/msrb/nvdlib
pushd nvdlib
pip install -r requirements.txt
python setup.py install
popd

# install the project
pip install -r requirements.txt

# download nltk data
python -c "import nltk; nltk.download('words')"
python -c "import nltk; nltk.download('stopwords')"
python -c "import nltk; nltk.download('wordnet')"
python -c "import nltk; nltk.download('universal_tagset')"
python -c "import nltk; nltk.download('averaged_perceptron_tagger')"

# ensure pytest and coverage is available
pip install pytest pytest-cov codecov

# run tests
PYTHONPATH=src/ pytest --cov="src/" --cov-report term-missing --cov-fail-under=$COVERAGE_THRESHOLD -vv tests/

codecov --token=e64bad60-3ce8-4089-97d0-5004cda9e1ce
