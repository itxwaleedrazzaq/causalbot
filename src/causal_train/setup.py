from setuptools import find_packages, setup

package_name = 'causal_train'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='waleed',
    maintainer_email='itxwaleedrazzaq@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'data_recorder = causal_test.data_recorder:main',
            'visualize_data = causal_test.visualize_data:main',
        ],
    },
)
