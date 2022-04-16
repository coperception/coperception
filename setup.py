from distutils.core import setup
setup(
  name = 'coperception',
  packages = ['coperception'],
  version = '0.0.1',
  license='apache-2.0',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'A library for collaborative perception.',   # Give a short description about your library
  author = 'AI4CE Lab @NYU',                   # Type in your name
  author_email = 'dm4524@nyu.edu',      # Type in your E-Mail
  url = 'https://ai4ce.github.io/',   # Provide either the link to your github or to your website
#   download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['computer-vision', 'deep-learning', 'autonomous-driving', 'collaborative-learning', 'knowledge-distillation', 'communication-networks', 'multi-agent-learning', 'multi-agent-system', '3d-object-detection', 'graph-learning', 'point-cloud-processing', 'v2x-communication', 'multi-agent-perception', '3d-scene-understanding'],   # Keywords that define your package best
  install_requires=[           
          'numpy',
          'torch',
          'opencv-python',
          'torchvision',
          'functools',
          'typing',
          'nuscenes-devkit',
          'pyquaternion',
          'numba',
          'matplotlib',
          'mmcv',
          'terminaltables',
          'shapely'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Libraries',
    'License :: OSI Approved :: Apache Software License',  
    'Programming Language :: Python :: 3.7',      #Specify which pyhton versions that you want to support
  ],
)