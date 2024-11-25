import sys
from pathlib import Path

from streamlit.web import cli as stcli


def main_run():
    app_path = Path(__file__).parent / "app.py"
    sys.argv = ["streamlit", "run", str(app_path)] + sys.argv[1:]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main_run()
