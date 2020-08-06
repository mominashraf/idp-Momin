import os
import glob
import argparse

from preparation import Preparation


def parseArguments(args=None):

    """
    parse command line arguments
    """

    # parse configuration
    parser = argparse.ArgumentParser(description='cowc data prep')
    parser.add_argument('footprint_pathname', action="store")
    parser.add_argument('data_path', action="store")
    parser.add_argument('out_path', action="store")

    return parser.parse_args(args)


def main():

    """
    main path of execution
    """

    # parse arguments
    args = parseArguments()
    
    # create image and annotations for keras-yolo3
    obj = Preparation( args.footprint_pathname )

    # locate all images in data path
    files = glob.glob( os.path.join( args.data_path, '*.tif' ) )
    for f in files:

        # generate sub-images and annotation
        # obj.process( f, args.out_path, aoi=[ 0, 0, 16000, 18000 ], writeback=False )
        obj.process( f, args.out_path, writeback=False )

    return


# execute main
if __name__ == '__main__':
    main()
