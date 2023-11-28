from setuptools import setup

package_name = 'cone_mapper'

setup(
    name=package_name,
    version='0.5.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kmhytting',
    maintainer_email='kmhytting@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mapping_node = cone_mapper.mapping_node:main',
        ],
    },
)
