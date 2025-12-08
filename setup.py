from setuptools import setup, find_packages

setup(
  name='movie_lens_reranker',
  version='0.1.0',
  packages=find_packages(where="src/main/python",
    include=['movie_lens_reranker']),
  package_dir={'': 'src/main/python'},
  install_requires = [
    'tensorflow-cpu>=2.20.0', 
  ],
  extras_require={"test": ["pytest"]},
  classifiers=[ 'Natural Language :: English',
               "Programming Language :: Python :: 3",
               'Programming Language :: Python :: 3.11',
               'Programming Language :: Python :: 3.12',
               'Programming Language :: Python :: 3.13',
               "Programming Language :: Python :: 3 :: Only",
               'Development Status :: 1 - Development/Unstable'
  ],
  url='https://www.kaggle.com/code/nicholeasuniquename/recommender-systems/',
  license='MIT',
  author='Nichole King',
  author_email='',
  description='Reranker for Kaggle recommender systems project'
)
