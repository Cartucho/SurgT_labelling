import argparse
from code.utils import load_yaml_data
from code.label import label_data


def main():
    parser = argparse.ArgumentParser(description='Tool for labeling stereo matches')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    config = load_yaml_data(args.config)
    label_data(config)


if __name__ == "__main__":
    main()
