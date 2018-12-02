from distutils.core import setup

setup(name='PaveAnalyser',
      version='0.1',
      description='Tool for analysing termal data measured during paving operations.',
      author='Lasse Grinderslev Andersen',
      author_email='lasse@etableret.dk',
      url='https://github.com/lgandersen/pave-analyser',
      packages=['pave_analyser'],
      entry_points = {
          'console_scripts': ['pave-analyser=pave_analyser.cli:script'],
          }
     )
