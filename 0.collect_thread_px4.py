import argparse
import os
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Personal information')
    parser.add_argument('--thread', dest='thread', type=int, help='Name of the candidate', default=6)
    args = parser.parse_args()
    thread = args.thread
    thread = int(thread)
    print(thread)

    for i in range(thread):
        time.sleep(1)
        cmd = f'gnome-terminal --tab --working-directory={os.getcwd()} -e ' \
              f'"python3 {os.getcwd()}/0.collect_px4.py --device {i}"'
        os.system(cmd)