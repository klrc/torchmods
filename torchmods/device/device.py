import os

from . import ssh


class _DeviceInterface():
    """ Communication with device via ssh tunnel.
    """

    def __init__(self, key):
        self.__key = key

    def exec_command(self, root, cmd):
        with ssh.open(self.__key) as f:
            out = f.cd(root).call(cmd)
        return out

    def push(self, localpath, remotepath):
        with ssh.open(self.__key) as f:
            print(localpath, remotepath)
            f.upload(localpath, remotepath)
        return self

    def pull(self, remotepath, localpath):
        with ssh.open(self.__key) as f:
            f.download(remotepath, localpath)
        return self


class _DeviceEnvironmentBlock(_DeviceInterface):
    """ A block of occupied device with full I/O access.
    """

    def __init__(self, key, root):
        super().__init__(key)
        with ssh.open(key) as f:
            f.call(f'[ -d {root} ] || mkdir {root}')
        self.__root = root

    def exec_command(self, cmd):
        return super().exec_command(self.__root, cmd)

    def cat1stline(self, filename):
        out = self.exec_command('cat ' + filename)
        out = out[0].replace('\n', '')
        return out

    def clear(self):
        super().exec_command('.', f'rm -r {self.__root}/*')
        return self

    def push(self, localpath):
        filename = localpath.split('/')[-1]
        super().push(localpath, self.__root + '/' + filename)
        return self

    def pull(self, filepath, localroot):
        filename = filepath.split('/')[-1]
        super().pull(self.__root + '/' + filepath, localroot + '/' + filename)
        return self

    def pull_all(self, localroot):
        for filename in self.exec_command('ls'):
            filename = filename.replace('\n', '')
            self.pull(filename, localroot)
        return self

    def path(self):
        # used for generating local mirror dir,
        # therefore we check if local path exists first.
        if not os.path.exists(self.__root):
            os.makedirs(self.__root)
        return self.__root


class _IOCache():
    """ Feed source files,
        ** Device, do yours, **
        Get packed outputs.
    """

    def __init__(self, key, root):
        self.__input = _DeviceEnvironmentBlock(key, root + '/input')
        self.__output = _DeviceEnvironmentBlock(key, root + '/output')

    def push(self, tar):
        self.__input.push(tar)
        return self

    def pull(self):
        mirror_dir = self.__output.path()
        self.__output.pull_all(mirror_dir)
        return self


class _BackServer():
    """ A handler to wake/control the offline server. (which load tasks and run infinitely)
    """

    def __init__(self, key, root, server_script=None):
        self.__core = _DeviceEnvironmentBlock(key, root + '/server')
        self.server_script = server_script

    def awake(self):
        return bool(self.__core.cat1stline('awake'))

    def wake(self):
        if self.awake():
            return
        if self.server_script is None:
            path = self.default_server_script()
        self.__core.push(path)
        self.__core.exec_command('source server.sh > history')

    def default_server_script(self):
        return os.path.dirname(os.path.abspath(__file__)) + '/server.sh'


class Device(_DeviceEnvironmentBlock):
    """ Consider remote CUDA server as just DEVICE.
        push task, wait for output.
        maybe check the status?
        we dont need anything else actually.
    """

    def __init__(self, key, root):
        super().__init__(key, root)
        self.io = _IOCache(key, root)
        self.core = _BackServer(key, root)

    def push(self, tid, source):
        tar = self.build(tid, source)
        self.io.push(tar)
        self.core.wake()
        return self

    def pull(self):
        self.io.pull()
        return self

    def check_something(self):
        # if we really need this.
        return


def mount(key, root):
    return Device(key, root)
