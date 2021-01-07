

import paramiko
import base64


class Tunnel():
    def __init__(self, key):
        self.__key = key
        self.__client = None
        self.__root = '~'

    @staticmethod
    def decode(key: str):
        sequence = base64.b64decode(key)
        return str(sequence, 'utf8').split('$')

    @staticmethod
    def connect(key):
        ssh = paramiko.SSHClient()  # 创建sshclient
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # 指定当对方主机没有本机公钥的情况时应该怎么办，AutoAddPolicy表示自动在对方主机保存下本机的秘钥
        ssh.connect(*Tunnel.decode(key))
        return ssh

    @staticmethod
    def sftp_callback(transferred, total):
        print(f'\rtransferred/total: {transferred}/{total} ({transferred/total:%})', end='')

    def __enter__(self):
        self.__client = self.connect(self.__key)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__client.close()

    def __call__(self, cmd):
        _, stdout, stderr = self.__client.exec_command(f'. /etc/profile;. .bashrc;cd {self.__root};{cmd}')
        out = stdout.readlines()
        err = stderr.readlines()
        if len(err) > 0:
            print(err[0])
            raise err[0]
        return out

    def upload(self, local, remote):
        with self.__client.open_sftp() as sftp:
            sftp.put(local, remote, callback=self.sftp_callback)
        print()
        return self

    def download(self, remote, local):
        with self.__client.open_sftp() as sftp:
            sftp.get(remote, local, callback=self.sftp_callback)
        print()
        return self

    def cd(self, arg1):
        self.call('cd ' + arg1)
        self.__root = arg1
        self.__root = self.call('pwd')[0].replace('\n', '')
        return self

    def call(self, command):
        return self.__call__(command)


def keygen(ip_address, user, password, port=22):
    sequence = f'{ip_address}${port}${user}${password}'
    return str(base64.b64encode(sequence.encode('utf8')), 'utf8')


def open(key):
    return Tunnel(key)
