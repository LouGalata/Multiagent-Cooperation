from setuptools import setup, find_packages

setup(name='multiagent',
      version='0.0.1',
      description='Multi-Agent through Graphs Cooperation',
      url='',
      author='Galata Elli',
      author_email='lougalata@gmail.com',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym', 'numpy-stl', 'keras']
)