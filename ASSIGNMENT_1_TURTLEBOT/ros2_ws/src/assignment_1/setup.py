from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'assignment_1'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=[
        'setuptools',
        'scikit-image',
    ],
    zip_safe=True,
    maintainer='arinweling',
    maintainer_email='you@example.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'astar_planner = assignment_1.astar_planner:main',
            'gvd_planner = assignment_1.gvd_planner:main',
            'path_follower = assignment_1.path_follower:main',
            'planner_server = assignment_1.planner_server:main',
        ],
    },
)
