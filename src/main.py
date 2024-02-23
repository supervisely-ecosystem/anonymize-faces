import supervisely as sly

from utils import main as run


def main():
    run()


if __name__ == "__main__":
    sly.main_wrapper("main", main)
