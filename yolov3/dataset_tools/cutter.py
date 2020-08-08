import argparse
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


parser = argparse.ArgumentParser()
parser.add_argument('FILENAME', help="Input file", type=str)
parser.add_argument('START', help='Start point (seconds)')
parser.add_argument('END', help='End point (seconds)')
args = parser.parse_args()

filename = args.FILENAME
t1 = int(args.START)
t2 = int(args.END)
target = '{}_{}_{}.mp4'.format(filename, t1, t2)

ffmpeg_extract_subclip(filename=filename,
                       t1=t1,
                       t2=t2,
                       targetname=target)
