import platform
import pkg_resources
import imp
def __bootstrap__():
    global __bootstrap__, __loader__, __file__
   
    supported_systems = {
        'Windows': 'mapping.cp37-win_amd64.pyd',
        'Linux': 'mapping.cpython-37m-x86_64-linux-gnu.so'
    }
    
    current_system = platform.system()
    if current_system not in supported_systems:
        raise NotImplementedError(f'Your system : {current_system}, is currently not supported. Please navigate to coperception/utils/mapping/dynamic_library to manually build the mapping shared library file.')

    plugin_name = supported_systems[current_system]
    __file__ = pkg_resources.resource_filename(
        __name__, plugin_name
    )
    __loader__ = None
    del __bootstrap__, __loader__
    imp.load_dynamic(__name__, __file__)


__bootstrap__()