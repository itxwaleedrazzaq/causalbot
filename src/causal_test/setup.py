from setuptools import find_packages, setup

package_name = 'causal_test'

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
            'gracam_visualize = causal_test.gracam_visualize:main',
            'test_ai = causal_test.test_ai:main',
            'record_dcm = causal_test.test_dcm_coeff:main',  
        ],
    },
)
    