#!D:\Documents\GitHub\OpenMDAO\venv_py3\Scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'openmdao','console_scripts','wingproj'
__requires__ = 'openmdao'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('openmdao', 'console_scripts', 'wingproj')()
    )
