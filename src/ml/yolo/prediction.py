import os
import osr
import ogr
import gdal
import json
import numpy as np
import matplotlib.patches as patches

from keras.models import load_model
from projects.yolo3.utils.utils import get_yolo_boxes


class Prediction:

    def __init__( self, model_pathname, out_pathname, prj ):

        """
        constructor
        """

        # load model weights from file
        self._model = load_model( model_pathname )

        # set parameters
        self._net_h, self._net_w = 416, 416 # a multiple of 32, the smaller the faster
        self._obj_thresh, self._nms_thresh = 0.5, 0.45

        # get yolo3 configuration settings
        config_pathname = os.path.join( os.path.dirname( model_pathname ), 'config.json' )
        with open( config_pathname ) as buffer:    
            self._config = json.load( buffer )   

        # get shapefile driver
        driver = ogr.GetDriverByName('Esri Shapefile')
        if os.path.exists( out_pathname ):
            driver.DeleteDataSource( out_pathname )

        # recreate outfile
        self._ds = driver.CreateDataSource( out_pathname )        
        if self._ds is not None:
    
            # create polygon layer
            self._srs = osr.SpatialReference( wkt=prj )
            self._layer = self._ds.CreateLayer('', self._srs, ogr.wkbPolygon)
            
            # create id field
            self._layer.CreateField(ogr.FieldDefn( 'id', ogr.OFTInteger ) )
            self._defn = self._layer.GetLayerDefn()

            # feature index counter
            self._idx = 1

        return


    def __del__( self ):

        """
        destructor
        """

        # delete variables to force write 
        self._layer = None
        self._ds = None

        return


    def process( self, image_pathname, writeback=False ):

        """
        compute inference for image and add detections to output shape file
        """

        # load image into array with gdal to access geolocation metadata
        ds = gdal.Open( image_pathname )
        image = ds.ReadAsArray()
        
        # transpose and flip to cv2 format
        image = np.transpose ( image, (1,2,0) )
        image = np.flip( image[:,:,0:3], axis=2 )

        height, width, channels = image.shape

        # get predicted bounding boxes
        bboxes = get_yolo_boxes( self._model, [image], self._net_h, self._net_w, self._config['model']['anchors'], self._obj_thresh, self._nms_thresh)[0]
        for bbox in bboxes:

            if bbox.classes[0] > self._obj_thresh:
            
                # ignore objects occluded by image edges
                if bbox.xmin > 0 and bbox.ymin > 0 and bbox.xmax < width - 1 and bbox.ymax < height - 1:

                    # convert polyline to geometry
                    wkt = self.getGeometry( bbox, ds.GetGeoTransform() ) 

                    # create new polygon feature
                    feature = ogr.Feature( self._defn )
                    feature.SetField( 'id', self._idx )
                    feature.SetGeometry( ogr.CreateGeometryFromWkt( wkt ) )

                    # add feature
                    self._layer.CreateFeature(feature)
                    self._idx += 1

                    feature = None

        return


    def getGeometry( self, bbox, transform ):

        """
        convert bbox to geometry registered in world coordinates
        """

        def image2World( img_x, img_y ):
            
            # convert pixel location to world coordinates
            geo_x = ( img_x * transform[ 1 ] ) + transform[ 0 ]
            geo_y = ( img_y * transform[ 5 ] ) + transform[ 3 ]

            return geo_x, geo_y

        # create geometry and add box nodes in world coordinates
        ring = ogr.Geometry(ogr.wkbLinearRing)

        pts = [ [ bbox.xmin, bbox.ymin ],
                [ bbox.xmax, bbox.ymin ],
                [ bbox.xmax, bbox.ymax ],
                [ bbox.xmin, bbox.ymax ]
         ]

        for pt in pts:
            x, y = image2World( pt[ 0 ], pt[ 1 ] )
            ring.AddPoint( x, y )

        # create polygon and return wkt
        poly = ogr.Geometry(ogr.wkbPolygon)        
        poly.AddGeometry(ring)

        return poly.ExportToWkt()
