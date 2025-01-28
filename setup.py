from setuptools import setup

setup(
    name="codealign",
    version="0.0.1",
    install_requires=['tree_sitter==0.23.1', 'z3-solver', 'tree-sitter-python==0.21.0', 'tree-sitter-c==0.23.1'],
    packages=['codealign', 'codealign.lang'],
    package_data={"codealign.lang": ["lang.so"]}
)
