

from torchmods.device import mount

if __name__ == "__main__":
    dev = mount('MTcyLjIwLjQ2LjIzNSQyMiRzaCRzaA==', './.device')
    print(dev.exec_command('ls'))
    # print(dev.cat1stline('log'))
