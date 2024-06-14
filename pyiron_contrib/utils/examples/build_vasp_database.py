from pyiron_contrib.utils.vasp import DatabaseGenerator
import argparse

def main():
    parser = argparse.ArgumentParser(description='Find and compress directories based on specified criteria.')
    parser.add_argument('directory', metavar='DIR', type=str, help='the directory to operate on')
    args = parser.parse_args()
    
    datagen = DatabaseGenerator(args.directory)
    df = datagen.build_database()

if __name__ == '__main__':
    main()
