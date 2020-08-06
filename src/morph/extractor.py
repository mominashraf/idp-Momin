import os
import cv2
import osr
import gdal, ogr
import numpy as np
import random
import imageio
import argparse
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from skimage import data, img_as_float
from skimage.segmentation import ( morphological_chan_vese, circle_level_set )


class FootprintExtractor:

    def __init__( self, object_size=100 ):

        """
        constructor
        """
            
        # largest object size in pixels
        self._object_size = object_size
        self._object_halfsize = int( self._object_size / 2 )

        # for animation
        self._images = []

        return


    def process( self, args, sigma=0.8, iterations=25, writeback=False ):

        """
        use expansive active contours to transform point geometries into best-fit boundary polygons demarking object footprints
        """

        # open centroid shape file
        centroid = self.openCentroidFile( args.centroid_pathname )

        # open image file
        image = self.openImageFile( args.image_pathname )

        # create footprint shape file
        footprint = self.createOutputFile( args.out_pathname, image )

        # check validity of files
        if centroid is not None and image is not None and footprint is not None:

            # convert centroid locations to pixel coordinates
            coords = self.getCentroidImageCoordinates( centroid, image )
            # random.shuffle( coords )
    
            # for each x, y centroid location
            for idx, coord in enumerate( coords ):

                # check valid sub-image                    
                if coord[ 0 ] + self._object_size < image[ 'ds' ].RasterXSize and coord[ 1 ] + self._object_size < image[ 'ds' ].RasterYSize:

                    # extract sub-image - check for error
                    sub_image = image[ 'band' ].ReadAsArray( coord[ 0 ], coord[ 1 ], self._object_size, self._object_size )
                    sub_image = gaussian_filter( sub_image, sigma=sigma )
                    sub_image = sub_image / ( 2^16 - 1) 
            
                    # define initial state of active contour 
                    init_ls = circle_level_set( sub_image.shape, (self._object_halfsize, self._object_halfsize), 5)

                    if writeback is True:

                        # callback for animation
                        ls = morphological_chan_vese(   sub_image, 
                                                        iterations=iterations,
                                                        init_level_set=init_ls,
                                                        smoothing=1, lambda1=1, lambda2=1,
                                                        iter_callback=self.visualCallback( sub_image ))

                    else:

                        # callback for animation
                        ls = morphological_chan_vese(   sub_image, 
                                                        iterations=iterations,
                                                        init_level_set=init_ls,
                                                        smoothing=1, lambda1=1, lambda2=1 )

                    # compute simplified polygon 
                    polyline = self.getPolyline( ls )

                    # convert polyline to geometry
                    wkt = self.getGeometry( polyline, coord, image[ 'transform'] ) 

                    # create new polygon feature
                    feature = ogr.Feature( footprint[ 'defn' ] )
                    feature.SetField( 'id', idx )
                    feature.SetGeometry( ogr.CreateGeometryFromWkt( wkt ) )

                    # add feature
                    footprint[ 'layer' ].CreateFeature(feature)
                    feature = None

                    # create animated gif
                    if idx == 5 and writeback:
                        imageio.mimsave( 'C:\\Users\\Chris.Williams\\Desktop\\animate.gif', self._images, fps=5, palettesize=64, subrectangles=True )
                        break
        
            # delete variables to force write 
            footprint[ 'layer'] = None
            footprint[ 'ds'] = None

        # buffer up footprint polygons by 3 metres
        self.getBufferedPolygons( args.out_pathname, 3 )
    
        return


    def openCentroidFile( self, pathname ):

        """
        load point geometries identifying object centroid locations 
        """
        
        # open centroid file
        centroid = None
        fid = ogr.Open( pathname )
        if fid is not None:

            # create dictionary with layer and srs info
            centroid = {    'fid': fid,
                            'layer' : fid.GetLayer(),
                            'srs' : fid.GetLayer().GetSpatialRef() }
            
        return centroid


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
        footprint = None

        driver = ogr.GetDriverByName('Esri Shapefile')
        if os.path.exists( pathname ):
            driver.DeleteDataSource( pathname )

        # create shapefile
        ds = driver.CreateDataSource( pathname )
        if ds is not None:
    
            # add row number id attribute
            srs = osr.SpatialReference(wkt=image[ 'projection' ] )

            layer = ds.CreateLayer('', srs, ogr.wkbPolygon)
            layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))

            # package up details into dictionary
            footprint = {   'ds': ds,
                            'layer' : layer,
                            'srs' : srs,
                            'defn' : layer.GetLayerDefn() }

        return footprint


    def getCentroidImageCoordinates( self, centroid, image ):

        """
        transform centroid in world coordinates to pixel location
        """

        # get transformation between image and centroid locations
        image_srs = osr.SpatialReference(wkt=image[ 'projection' ] )
        tx = osr.CoordinateTransformation ( centroid[ 'srs' ], image_srs )

        # for each point geometry
        coords = []
        for feature in centroid[ 'layer' ]:

            # transform to image srs
            geom = feature.GetGeometryRef()
            (ulx, uly, ulz ) = tx.TransformPoint( geom.GetX(), geom.GetY() )

            # convert geo position to image location
            col = (( ulx - image[ 'transform' ][ 0 ] ) / image[ 'transform' ][ 1 ] ) 
            row = (( uly - image[ 'transform' ][ 3 ] ) / image[ 'transform' ][ 5 ] )

            # create list of sub-image origins
            coords.append( [ max( round ( col - self._object_halfsize ), 0 ), 
                                max( round ( row - self._object_halfsize ), 0 ) ] )
            
        return coords


    def getPolyline( self, ls ):

        """
        smooth and simplify boundary of active contour
        """

        # get the max-area contour
        cnts = cv2.findContours( ls.astype( np.uint8 ), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2]    
        cnt = sorted(cnts, key=cv2.contourArea)[-1]

        # calculate arc length
        arclen = cv2.arcLength(cnt, True)

        eps = 0.0005
        epsilon = arclen * eps
        
        # compute simplied polyline
        approx = cv2.approxPolyDP( cnt, epsilon, True)
        approx = np.vstack( ( approx, np.expand_dims( approx[ 0 ], axis=0 ) ) )

        return np.vstack( ( approx[ :, 0, 0 ], approx[ :, 0, 1 ] ) ).T


    def getGeometry( self, polyline, coord, transform ):

        """
        transform polyline from image to geo coordinates
        """

        # transform polyline from image to geo coordinates
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for point in polyline:

            geo_x = ( ( coord[ 0 ] + point[ 0 ] ) * transform[ 1 ] ) + transform[ 0 ]
            geo_y = ( ( coord[ 1 ] + point[ 1 ] ) * transform[ 5 ] ) + transform[ 3 ]

            ring.AddPoint( geo_x, geo_y )

        # create polygon and return wkt
        poly = ogr.Geometry(ogr.wkbPolygon)        
        poly.AddGeometry(ring)

        return poly.ExportToWkt()


    def getBufferedPolygons( self, footprint_pathname, distance ):
    
        """
        apply buffer distance to polygon geometries 
        """

        # open polygon shapefile
        ds = ogr.Open( footprint_pathname )
        if ds is not None:

            # create buffered polygon shapefile
            layer = ds.GetLayer()
            buffer = self.createBufferFile( footprint_pathname, layer.GetSpatialRef() )

            if buffer is not None:

                # for each feature
                for feature in layer:
                    
                    # create buffered polygon
                    geom = feature.GetGeometryRef()
                    buffer_geom = geom.Buffer( distance )

                    # add buffered polygon to output file
                    feature = ogr.Feature( buffer[ 'defn' ] )
                    feature.SetGeometry( buffer_geom )
                    
                    buffer[ 'layer' ].CreateFeature( feature )
                    feature = None

        return


    def createBufferFile( self, pathname, sr ):

        """
        write buffered footprint polygons to output vector file
        """

        # create output filename
        out_pathname = pathname.replace( '.shp', '_buffer.shp' )
        buffer = {}

        # delete if exists
        driver = ogr.GetDriverByName('ESRI Shapefile')
        if os.path.exists( out_pathname ):
            driver.DeleteDataSource( out_pathname )

        # create data source
        ds = driver.CreateDataSource( out_pathname )
        if ds is not None:

                # package up buffer file properties into dictionary
                layer = ds.CreateLayer('', sr, ogr.wkbPolygon)
                buffer = {  'ds': ds,
                            'layer' : layer,
                            'defn' : layer.GetLayerDefn() }

        return buffer


    def visualCallback( self, background, fig=None):

        """
        Returns a callback than can be passed as the argument `iter_callback`
        of `morphological_geodesic_active_contour` and
        `morphological_chan_vese` for visualizing the evolution
        of the levelsets. Only works for 2D images.
        
        Parameters
        ----------
        background : (M, N) array
            Image to be plotted as the background of the visual evolution.
        fig : matplotlib.figure.Figure
            Figure where results will be drawn. If not given, a new figure
            will be created.
        
        Returns
        -------
        callback : Python function
            A function that receives a levelset and updates the current plot
            accordingly. This can be passed as the `iter_callback` argument of
            `morphological_geodesic_active_contour` and
            `morphological_chan_vese`.
        
        """
        
        # Prepare the visual environment.
        if fig is None:
            fig = plt.figure()
        fig.clf()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(background, cmap=plt.cm.gray)

        ax2 = fig.add_subplot(1, 2, 2)
        ax_u = ax2.imshow(np.zeros_like(background), vmin=0, vmax=1)
        fig.tight_layout()
        plt.pause(0.001)

        def callback(levelset):
            
            if ax1.collections:
                del ax1.collections[0]
            ax1.contour(levelset, [0.5], colors='r')
            ax_u.set_data(levelset)
            fig.canvas.draw()

            # cache renderer
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            self._images.append( cv2.resize( image, dsize=(320,240 ) ) )

            plt.pause(0.001)

        return callback


def parseArguments( args=None ):

    """
    parse command line arguments
    """

    # parse configuration
    parser = argparse.ArgumentParser(description='centroid localiser')
    parser.add_argument('centroid_pathname', action="store")
    parser.add_argument('image_pathname', action="store")
    parser.add_argument('out_pathname', action="store")

    return parser.parse_args(args)


# execute main
if __name__ == '__main__':

    # parse arguments
    args = parseArguments()

    # execute
    obj = FootprintExtractor()
    obj.process( args ) 
