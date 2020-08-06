import os
import cv2
import osr
import ogr
import json
import argparse

from shapely.geometry import shape
import matplotlib.pyplot as plt


class FootprintFilter:


    def __init__( self ):

        """
        constructor
        """

        return


    def process( self, args, writeback=False ):

        """
        filter footprints based on spatial attributes
        """

        # open footprint file
        footprint = self.openFootprintFile( args.footprint_pathname )
        output = self.createOutputFile( args.out_pathname, footprint[ 'srs' ].ExportToWkt() )

        for feature in footprint[ 'layer' ]:

            # convert ogr feature to shapely object
            json_obj = json.loads( feature.ExportToJson() )
            geom = shape( json_obj['geometry' ])

            print ( ( geom.area / geom.minimum_rotated_rectangle.area ) * 100 )

            if geom.area / geom.minimum_rotated_rectangle.area > 0.60:

                # add feature
                output[ 'layer' ].CreateFeature(feature)

        return


    def openFootprintFile( self, pathname ):

        """
        load footprint geometries from shapefile
        """
        
        # open centroid file
        footprint = None
        fid = ogr.Open( pathname )
        if fid is not None:

            # create dictionary with layer and srs info
            footprint = {   'fid': fid,
                            'layer' : fid.GetLayer(),
                            'srs' : fid.GetLayer().GetSpatialRef() }
            
        return footprint


    def createOutputFile( self, pathname, wkt ):

        """
        create output file to store filtered footprints
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
            srs = osr.SpatialReference( wkt=wkt )

            layer = ds.CreateLayer('', srs, ogr.wkbPolygon)
            layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))

            # package up details into dictionary
            footprint = {   'ds': ds,
                            'layer' : layer,
                            'srs' : srs,
                            'defn' : layer.GetLayerDefn() }

        return footprint


def parseArguments( args=None ):

    """
    parse command line arguments
    """

    # parse configuration
    parser = argparse.ArgumentParser(description='footprint filter')
    parser.add_argument('footprint_pathname', action="store", help="pathname to footprint shapefile")
    parser.add_argument('out_pathname', action="store", help="output filter footprint shapefile")

    return parser.parse_args(args)


# execute main
if __name__ == '__main__':

    # parse arguments
    args = parseArguments()

    # execute
    obj = FootprintFilter()
    obj.process( args ) 
