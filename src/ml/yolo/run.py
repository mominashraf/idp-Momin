import os
import gdal
import glob
import argparse
import geopandas as gpd

from prediction import Prediction

def parseArguments(args=None):

    """
    parse command line arguments
    """

    # parse configuration
    parser = argparse.ArgumentParser(description='prediction')
    parser.add_argument('model_pathname', action="store")
    parser.add_argument('data_path', action="store")
    parser.add_argument('out_pathname', action="store")

    return parser.parse_args(args)


def main():

    """
    main path of execution
    """

    # parse arguments
    args = parseArguments()

    # locate jpeg training images
    files = glob.glob( os.path.join( args.data_path, 'images/*.jpg' ) )
    if len( files ) > 0:

        # open first image to extract projection
        ds = gdal.Open( files[ 0 ] )
        if ds is not None:

            # create prediction object
            obj = Prediction( args.model_pathname, args.out_pathname, ds.GetProjection() )

            # compute inference and add detections to output shape file
            for f in files:
                obj.process( f, writeback=False )


    # read shapefile
    gdf = gpd.read_file( args.out_pathname )

    # locate isolated features 
    bad_rows = []
    for idx, row in gdf.iterrows():
        if sum( gdf['geometry'].buffer(100).intersects( row['geometry'] ) ) < 5:
            bad_rows.append( idx )

    # drop isolated features 
    gdf = gdf.drop( bad_rows )
    gdf.to_file( args.out_pathname.replace( '.shp', '-update.shp' ) )

    return


# execute main
if __name__ == '__main__':
    main()
