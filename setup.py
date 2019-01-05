from distutils.core import setup

setup(name='RoadTherma',
      version='0.1',
      description='Tool for analysing termal data measured during paving operations.',
      author='Lasse Grinderslev Andersen',
      author_email='lasse@etableret.dk',
      url='https://github.com/roadtools/roadtherma',
      packages=['roadtherma'],
      entry_points = {
          'console_scripts': ['roadtherma=roadtherma.cli:script'],
          }
     )
