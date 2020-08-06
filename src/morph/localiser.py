import os
import cv2
import osr
import gdal, ogr
import argparse
import numpy as np

from skimage import io
from skimage.measure import label
from shapely.geometry import Point

import matplotlib.pyplot as plt


class CentroidLocaliser:


    def __init__( self ):

        """
        constructor
        """

        # maximum valid size of label blob 
        self._maxBlobSize = 1000
        return


    def process( self, args, writeback=False ):

        """
        get dataset handle and geolocation metadata 
        """

        # open image file
        image = self.openImageFile( args.image_pathname )
        data = image[ 'band' ].ReadAsArray()

        # create output shapefile
        centroids = self.createOutputFile( args.out_pathname, image )
        
        # generate threshold image
        idx = data > args.threshold
        mask = np.zeros_like( data )
        mask[ idx ] = 1

        # grab kernel and apply open operation
        kernel = cv2.getStructuringElement( cv2.MORPH_RECT, (5,5) )
        mask_open = cv2.morphologyEx( mask, cv2.MORPH_OPEN, kernel, iterations=args.iterations )

        # writeback
        if writeback:
            
            data[ np.where ( mask_open == 1 ) ] = 0
            plt.imshow( data, cmap='Greys' )
            plt.show()

        # label blob map
        labels = label( mask_open )
        for idx in range( 1, np.amax( labels ) + 1 ):

            # find location of blob with current label
            coords = np.where( labels == idx )
            if len ( coords[ 0 ] ) < self._maxBlobSize:
                
                # create shapely point object
                img_x = np.mean( coords[ 1 ] )
                img_y = np.mean( coords[ 0 ] )

                # compute geo coordinates
                geo_x = ( img_x * image[ 'transform' ][ 1 ] ) + image[ 'transform' ][ 0 ]
                geo_y = ( img_y * image[ 'transform' ][ 5 ] ) + image[ 'transform' ][ 3 ]

                # create new feature (attribute and geometry)
                feature = ogr.Feature( centroids[ 'defn' ] )
                feature.SetField( 'id', idx )

                # create geometry from shapely object
                geom = ogr.CreateGeometryFromWkb( Point( geo_x, geo_y ).to_wkb() )
                feature.SetGeometry(geom)

                centroids[ 'layer' ].CreateFeature( feature )

        return 


    def openImageFile( self, pathname, band_index=1 ):

        """
        load corresponding georeferenced raster 
        """

        # open image file
        image = None
        ds = gdal.Open( pathname )
        if ds is not None:

            # compile image information into dictionary
            image = {   'ds': ds,
                        'band' : ds.GetRasterBand( band_index ),
                        'transform' : ds.GetGeoTransform(),
                        'projection' : ds.GetProjection()
            }

        return image


    def createOutputFile( self, pathname, image ):

        """
        create output vector to write boundary footprint polygon geometries 
        """

        # overwrite existing file
        centroids = None

        driver = ogr.GetDriverByName('Esri Shapefile')
        if os.path.exists( pathname ):
            driver.DeleteDataSource( pathname )

        # create shapefile
        ds = driver.CreateDataSource( pathname )
        if ds is not None:
    
            # add row number id attribute
            srs = osr.SpatialReference(wkt=image[ 'projection' ] )

            layer = ds.CreateLayer('', srs, ogr.wkbPoint)
            layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))

            # package up details into dictionary
            centroids = {   'ds': ds,
                            'layer' : layer,
                            'srs' : srs,
                            'defn' : layer.GetLayerDefn() }

        return centroids


def parseArguments( args=None ):

    """
    parse command line arguments
    """

    # parse configuration
    parser = argparse.ArgumentParser(description='centroid localiser')
    parser.add_argument('image_pathname', action="store", help="pathname to image")
    parser.add_argument('out_pathname', action="store", help='output shapefile')

    parser.add_argument('-t', '--threshold', type=int, help='threshold', default='900')
    parser.add_argument('-i', '--iterations', type=int, help='iterations', default='1')

    return parser.parse_args(args)


# execute main
if __name__ == '__main__':

    # parse arguments
    args = parseArguments()

    # execute
    obj = CentroidLocaliser()
    obj.process( args ) 
