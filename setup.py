import os
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand

test_args = [
    '--verbose',
    '--capture=sys',
    '--log-level=INFO',
    '--log-cli-level=INFO',
    '--log-file-level=INFO',
    '--no-flaky-report',
    '--timeout=300',
    '--cov-report=html',
    '--cov-report=term',
    '--cov-report=xml',
    '--cov=torcharc',
    'test',
]


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', 'Arguments to pass to py.test')]

    def initialize_options(self):
        os.environ['PY_ENV'] = 'test'
        TestCommand.initialize_options(self)
        self.pytest_args = test_args

    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    name='torcharc',
    version='0.0.6',
    description='Build PyTorch networks by specifying architectures.',
    long_description='https://github.com/kengz/torcharc',
    keywords='torcharc',
    url='https://github.com/kengz/torcharc',
    author='kengz',
    author_email='kengzwl@gmail.com',
    packages=[
        'torcharc',
        'torcharc.module',
        'torcharc.module.transformer',
    ],
    install_requires=[
        'numpy>=1.16.1',
        'pydash==4.7.6',
    ],
    zip_safe=False,
    include_package_data=True,
    dependency_links=[],
    extras_require={},
    classifiers=[],
    tests_require=[
        'autopep8==1.5.3',
        'flake8==3.8.3',
        'flaky==3.6.1',
        'pytest==5.4.1',
        'pytest-cov==2.8.1',
        'pytest-timeout==1.3.4',
    ],
    test_suite='test',
    cmdclass={'test': PyTest},
)
