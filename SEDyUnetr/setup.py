# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from setuptools import setup, find_namespace_packages
from io import open


with open('requirements.txt', encoding="utf-8-sig") as f:
    requirements = f.readlines()

def readme():
    with open('readme_unetr.md', encoding="utf-8-sig") as f:
        README = f.read()
    return README


setup(
    name='DyUnetr',
    packages=find_namespace_packages(include=["DyUnetr", "DyUnetr.*"]),
    entry_points={"console_scripts": [
        "DyUnetr_main= DyUnetr.execute:main"
    ]},
    version='1.1.3',
    install_requires=requirements,
    license='Apache License 2.0',
    description='Dynamic Unetr for multiple organs',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='',
    download_url='',
    keywords=[
        'Unet, Transformer, Dynamic'
    ],
    classifiers=[
        'Intended Audience :: Developers', 'Operating System :: OS Independent',
        'Natural Language :: Chinese (Simplified)',
        'Programming Language :: Python :: 3.8'
    ] )
