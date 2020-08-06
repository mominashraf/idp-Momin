import os
import cv2
import osr
import gdal
import shutil
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import dicttoxml
from xml.dom.minidom import parseString

from collections import OrderedDict
from shapely.geometry import Polygon

from slicer import Slicer


class Preparation:

    def __init__( self, footprint_pathname ):

        """
        constructor
        """

        # read geometry pathname
        self._gdf = gpd.read_file( footprint_pathname )

        # create image slicer
        self._slicer = Slicer( width=128, height=128, overlap=0.4 )

        # xml conversion tweak
        self._custom_item_func = lambda x: 'object'

        return


    def process( self, image_pathname, out_path, aoi=None, writeback=False ):

        """
        create images and annotations for train and validation
        """

        # slice and dice
        slices = self._slicer.process ( image_pathname, 
                                        os.path.join( out_path, 'images' ),
                                        aoi=aoi,
                                        percentiles=self._slicer.getPercentiles( image_pathname ) )  

        # load master image
        src = self.loadImageDataset( image_pathname )
        if src is not None:

            # transform geodataframe to image crs
            sr = osr.SpatialReference( wkt=src[ 'projection' ] )
            gdf = self._gdf.to_crs( sr.ExportToProj4() )

            # for each slice
            records = []
            for s in slices:

                # determine intersection with footprint list and slice bounding box
                extent = self.getSliceExtent( s, src[ 'transform' ] )
                intersect_list = gdf[ gdf[ 'geometry' ].intersects( extent ) ]

                # something intersects
                results = []
                if not intersect_list.empty:

                    # get origin of slice in geo coordinates
                    origin = [ extent.bounds[ 0 ], extent.bounds[ 3 ] ]

                    # derive image bounding boxes for intersecting polygons
                    for idx, row in intersect_list.iterrows():                        

                        result = self.getBoundingBox( s, origin, row[ 'geometry'], src[ 'transform' ] )
                        if result:
                            results.append ( result )

                # create sub-directory for annotation pascal voc xml files
                annotation_path = os.path.join( out_path, 'annotations' )
                if not os.path.exists ( annotation_path ):
                    os.makedirs( annotation_path )

                # create schema and write annotation xml file
                self.getAnnotation( s, results, annotation_path, writeback )

                # record count and areal coverage 
                area = 0
                for result in results:
                    area += ( ( result[ 'bbox' ][ 2 ] - result[ 'bbox' ][ 0 ] ) * ( result[ 'bbox' ][ 3 ] - result[ 'bbox' ][ 1 ] ) )

                records.append( {   'pathname' : s[ 'pathname' ],
                                    'count' : len( results ),
                                    'area' : area,
                                    'x0' : s[ 'x0' ],
                                    'y0' : s[ 'y0' ] } )

            # save records as csv via data frame
            df = pd.DataFrame( records )
            df.to_csv( os.path.join( os.path.join( out_path, 'annotations' ), 'labels.csv' ) )

        return


    def loadImageDataset( self, pathname ):

        """
        get dataset handle and geolocation metadata 
        """

        # open image file
        image = None
        ds = gdal.Open( pathname )
        if ds is not None:

            # compile image information into dictionary
            image = {   'ds': ds,
                        'transform' : ds.GetGeoTransform(),
                        'projection' : ds.GetProjection()
            }

        return image


    def getSliceExtent( self, s, transform ):

        """
        get bounding box of image slice in georeferenced coordinates
        """

        # get geo coordinates of image slice
        x0 = ( s[ 'x0' ] * transform[ 1 ] ) + transform[ 0 ]
        y0 = ( s[ 'y0' ] * transform[ 5 ] ) + transform[ 3 ]
        x1 = ( ( s[ 'x0' ] + s[ 'width'] )  * transform[ 1 ] ) + transform[ 0 ]
        y1 = ( ( s[ 'y0' ] + s[ 'height'] ) * transform[ 5 ] ) + transform[ 3 ]

        return Polygon( [ (x0,y0), (x0,y1), (x1,y1), (x1,y0)] )


    def getBoundingBox( self, s, origin, polygon, transform ):

        """
        compute footprint bounding box relative to image slice
        """

        # get location and image sizes
        dims = ( s[ 'height' ], s[ 'width' ] )
        result = {} 

        # compute polygon bounding box in image coordinates
        x0 = ( polygon.bounds[ 0 ] - origin[ 0 ] ) / transform[ 1 ]
        y0 = ( polygon.bounds[ 1 ] - origin[ 1 ] ) / transform[ 5 ]
                        
        x1 = ( polygon.bounds[ 2 ] - origin[ 0 ] ) / transform[ 1 ]
        y1 = ( polygon.bounds[ 3 ] - origin[ 1 ] ) / transform[ 5 ]

        # get min and maxes
        x_min = min( x0, x1 ); x_max = max ( x0, x1 )
        y_min = min( y0, y1 ); y_max = max ( y0, y1 )

        # check limits
        x_min_c = max( 0, x_min ); y_min_c = max( 0, y_min )
        x_max_c = min( x_max, dims[1] - 1 ); y_max_c = min( y_max, dims[0] - 1 )        

        area = ( x_max - x_min ) * ( y_max - y_min )
        area_c = ( x_max_c - x_min_c ) * ( y_max_c - y_min_c )

        print ( 'coords {} {} {} {}'.format( x_min, y_min, x_max, y_max ) )
        print ( 'area {} {}'.format( area, area_c ) )

        # only retain bboxes not constrained by image edges
        if area_c / area > 0.95:
            result[ 'bbox' ] = [ x_min_c, y_min_c, x_max_c, y_max_c ]

        return result


    def getAnnotation( self, s, results, out_path, writeback, overwrite=True ):

        """
        create annotation xml files encoding bounding box locations
        """

        # create label pathname
        filename = os.path.splitext( os.path.basename( s[ 'pathname' ] ) )[ 0 ] + '.xml' 
        annotation_pathname = os.path.join( out_path, filename )

        if not os.path.exists( annotation_pathname ) or overwrite:

            schema = self.getSchema( s, results )

            # create output dir if necessary
            if not os.path.exists( out_path ):
                os.makedirs( out_path )

            # write annotation to xml file
            with open( annotation_pathname, "w+" ) as outfile:

                # parse xml into string
                xml = dicttoxml.dicttoxml( schema, attr_type=False, item_func=self._custom_item_func, custom_root='annotation' ) \
                        .replace(b'<annotation>',b'<annotation verified="yes">') \
                        .replace(b'<items>',b'').replace(b'</items>',b'') \

                dom = parseString( xml )

                # write xml string to file
                outfile.write( dom.toprettyxml() )

            # plot writeback
            if writeback:
                self.drawBoundingBoxes( s[ 'pathname' ], results )

        return

        
    def getSchema( self, s, records ):

        """
        convert annotation into ordered list for conversion into PASCAL VOC schema
        """

        # convert to PASCAL VOC annotation schema
        object_list = []
        for record in records:

            bbox = record[ 'bbox' ]; #corner = record[ 'corner' ]
            object_list.append( OrderedDict ( {     'name' : 'idp',
                                                    'pose': 'Topdown',
                                                    'truncated' : 0,
                                                    'difficult': 0,
                                                    'bndbox': {'xmin': bbox[ 0 ], 'ymin': bbox[ 1 ], 'xmax': bbox[ 2 ], 'ymax': bbox[ 3 ] }
                                                    #'segmentation' : ','.join( (str(pt) for pt in corner ) ) 
                                            } ) )

        # return full schema as dictionary
        return OrderedDict ( {  'folder' : 'images',
                                'filename' : os.path.basename( s[ 'pathname' ] ),
                                'path' : os.path.dirname( s[ 'pathname' ] ),
                                'source' : { 'database': 'null' },
                                'size' : { 'width' : s[ 'width' ], 'height' : s[ 'height' ], 'depth' : 3 },
                                'segmented' : 0,
                                'items' : object_list } )


    def drawBoundingBoxes( self, pathname, records ):

        """
        display image and draw bounding boxes for validation purposes
        """

        # no action if no bboxes
        if len ( records ) > 0:

            # load image
            img = cv2.imread( pathname )                                  
            height = img.shape[0]; width = img.shape[ 1 ]
                    
            # show image
            plt.imshow( cv2.cvtColor(img, cv2.COLOR_BGR2RGB) ); ax = plt.gca()
            fig = plt.gcf(); fig.canvas.set_window_title( os.path.basename( pathname ) )
            print ( pathname )

            # draw bbox lines
            colors = [ 'r', 'g', 'y', 'b', 'm', 'c' ]; idx = 0
            for record in records:    

                x0, y0, x1, y1 = record[ 'bbox' ]

                color = colors[ idx ] + '-'
                idx = idx + 1 if idx + 1 < len ( colors ) else 0

                ax.plot( [ x0, x1 ], [ y0, y0 ], color )
                ax.plot( [ x0, x1 ], [ y1, y1 ], color )
                ax.plot( [ x0, x0 ], [ y0, y1 ], color )
                ax.plot( [ x1, x1 ], [ y0, y1 ], color )

                """
                # get run length encoding from perimeter points string
                rl_encoding = mask.frPyObjects( [ record[ 'corner' ] ] , height, width )

                binary_mask = mask.decode( rl_encoding )
                binary_mask = np.amax(binary_mask, axis=2)

                masked = np.ma.masked_where(binary_mask == 0, binary_mask )
                ax.imshow( masked, 'jet', interpolation='None', alpha=0.5 )
                """

            plt.show()

        return
