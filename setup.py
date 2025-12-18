from setuptools import setup, find_packages

setup(
  name='movie_lens_reranker',
  version='0.1.0',
  packages=find_packages(where="src/main/python",
    include=['movie_lens_reranker' 'movie_lens_reranker.prompts']),
  package_dir={'': 'src/main/python'},
  install_requires = [
    'torch>=2.2',
    'tqdm>=4.67.1',
    'sentencepiece>=0.2.1',
    'transformers==4.57.3',
    'PyYAML>=6.0.3',
    'jinja2>=3.1.6',
    'datasets>=4.4.1',
    'ranx>=0.3.21',
    'peft>=0.18.0'
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
