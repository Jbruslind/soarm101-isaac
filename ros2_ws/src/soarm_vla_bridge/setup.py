from setuptools import find_packages, setup

package_name = "soarm_vla_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "vla_bridge = soarm_vla_bridge.vla_bridge_node:main",
        ],
    },
)
