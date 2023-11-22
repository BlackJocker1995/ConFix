import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Personal information')
    parser.add_argument('--thread', dest='thread', type=int, help='Name of the candidate', default=1)
    args = parser.parse_args()
    thread = args.thread
    thread = int(thread)
    print(thread)

    for i in range(thread):
        cmd = f'gnome-terminal --tab --working-directory={os.getcwd()} -e ' \
              f'"python3 {os.getcwd()}/0.collect.py --device {i}"'
        os.system(cmd)