#!D:\Documents\GitHub\OpenMDAO\venv_py3\Scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'testflo==1.3.4','console_scripts','testflo'
__requires__ = 'testflo==1.3.4'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('testflo==1.3.4', 'console_scripts', 'testflo')()
    )
