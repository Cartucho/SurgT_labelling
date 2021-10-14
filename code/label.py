class Interface:
    def __init__(self, config):
        print(config)

    def main_loop(self):
        """ Interface's main loop """
        print("Finished")


def label_data(config):
    print(config)
    inter = Interface(config)
    inter.main_loop()
