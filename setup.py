from setuptools import setup, find_packages

setup(
  name='movie_lens_reranker',
  version='0.1.0',
  packages=find_packages(where="src/main/python",
    include=['movie_lens_reranker' 'movie_lens_reranker.prompts']),
  package_dir={'': 'src/main/python'},
  install_requires = [
    'tensorflow-cpu>=2.19.0', 
    'tf_keras>=2.20.0',
    #'keras-hub>=0.24.0',
    'sentencepiece-0.2.1',
    'torch>=2.2',
    'transformers==4.57.3',
    'pyyaml>=6.03',
    'jinja2>=3.16'
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
