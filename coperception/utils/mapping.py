def __bootstrap__():
    global __bootstrap__, __loader__, __file__
    import shutil, os, sys, pkg_resources, imp
    import re

    current_dir = os.path.dirname(__file__)
    plugin_build_dir = current_dir + '/dynamic_library/build'
    if not os.path.exists(plugin_build_dir):
        os.makedirs(plugin_build_dir)
        os.system(f'cd {plugin_build_dir}&&cmake ..&&cmake --build .')
    plugin_path = None
    plugin_name = None
    for file in os.listdir(plugin_build_dir):
        if re.match('mapping.*\.(pyd|so)',file):
            plugin_name = file
            plugin_path = plugin_build_dir + '/' + plugin_name
            break
    shutil.copy(plugin_path, current_dir)
    __file__ = pkg_resources.resource_filename(
        __name__, plugin_name
    )
    __loader__ = None
    del __bootstrap__, __loader__
    imp.load_dynamic(__name__, __file__)


__bootstrap__()
