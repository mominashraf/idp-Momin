import os
import cv2
import gdal
import numpy as np

from shapely import geometry


class Slicer():

    def __init__( self, width=544, height=544, overlap=0.1 ):

        """
        constructor - setup default sub-image dimensions and overlap
        """

        # slicing variables
        self._overlap = overlap
        self._width = width
        self._height = height

        return


    def process( self, image_pathname, out_path, aoi=None, pad=0, percentiles=None, jpg_quality=95 ):

        """
        slice large image into smaller chunks and save to output path
        optional pixel-based area of interest and border padding
        """

        # read image
        image = self.loadDataset( image_pathname )
        slices = []

        # get size of image and slice size
        im_h, im_w = image[ 'ds' ].RasterYSize, image[ 'ds' ].RasterXSize
        win_size = self._height * self._width

        # initialise stride - maintain overlap between slices
        dx = int((1. - self._overlap) * self._width )
        dy = int((1. - self._overlap) * self._height )

        # 10% overlap between image slices
        for y in range(0, im_h, dy):
            for x in range(0, im_w, dx):

                # get new sub-image origin - careful not to past image edge
                y0 = ( im_h - self._height ) if y + self._height > im_h else y
                x0 = ( im_w - self._width ) if x + self._width > im_w else x

                if aoi is None or self.isIntersection( aoi, [ y0, x0, y0 + self._height, x0 + self._width ] ): 

                    # create output pathname
                    filename = 'slice_{}_{}_{}_{}_{}_{}'.format( os.path.splitext( os.path.basename( image_pathname ) )[0], 
                                                                y0, 
                                                                x0, 
                                                                self._height, 
                                                                self._width, 
                                                                pad )
                    out_pathname = os.path.join( out_path, filename + '.jpg')
                    if not os.path.exists( out_pathname ):

                        # create output folder in readiness
                        if not os.path.exists( out_path ):
                            os.makedirs( out_path )

                        # read slice as numpy array and switch dimensions
                        window_c = image[ 'ds' ].ReadAsArray( xoff=x0, yoff=y0, xsize=self._width, ysize=self._height )
                        window_c = np.transpose ( window_c, (1,2,0) )

                        # apply rescaling if non 8bit
                        if window_c.dtype != np.dtype('uint8'):

                            window_c = window_c.astype( float )
                            for idx in range( image[ 'ds' ].RasterCount ):

                                if percentiles is None:
                                    # get 2nd and 98th percentiles 
                                    r = np.percentile( window_c[ :,:,idx ], [ 2, 98 ] )
                                else:
                                    # use optional argument
                                    r = percentiles[ idx ]

                                window_c[ :,:,idx] = ( window_c[ :,:,idx ] - r[ 0 ] ) / ( r[ 1 ] - r[ 0 ] ) * 255.0

                            # convert back to byte
                            window_c[ window_c < 0.0 ] = 0.0; window_c[ window_c > 255.0 ] = 255.0
                            window_c = window_c.astype( np.uint8 )

                        # save jpeg using opencv due to colour space issues with gdal
                        window_c = np.flip( window_c[:,:,0:3], axis=2 )
                        cv2.imwrite( out_pathname, window_c, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality] )

                        # add geocoding
                        self.setSliceGeoTransform( out_pathname, image, x0, y0 )

                    # save sub-image as slice
                    slices.append( {    'pathname' : out_pathname, 
                                        'y0' : y0, 
                                        'x0' : x0, 
                                        'height' : self._height,
                                        'width' : self._width  } )

        return slices


    def loadDataset( self, pathname ):

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


    def setSliceGeoTransform( self, pathname, image, x0, y0 ):

        """
        compute slice-offset geotransform coefficients and update image
        """

        # open image file
        ds = gdal.Open( pathname )
        if ds is not None:

            # get geo coordinates of image slice
            geo_t = list( image[ 'transform' ] )

            geo_t[ 0 ] = ( x0 * geo_t[ 1 ] ) + geo_t[ 0 ]
            geo_t[ 3 ] = ( y0 * geo_t[ 5 ] ) + geo_t[ 3 ]

            ds.SetGeoTransform( geo_t )  
            ds.SetProjection( image[ 'projection' ] )                        

            ds = None
        
        return


    def isIntersection( self, r1, r2 ):

        """
        boolean check if rectangles overlap in cartesian space
        """

        # calculate intersection between aoi and slice window rectangles
        p1 = geometry.Polygon([(r1[0],r1[1]), (r1[1],r1[1]),(r1[2],r1[3]),(r1[2],r1[1])])
        p2 = geometry.Polygon([(r2[0],r2[1]), (r2[1],r2[1]),(r2[2],r2[3]),(r2[2],r2[1])])

        return(p1.intersects(p2))


    def getPercentiles( self, pathname, band_idxs=[1,2,3], nbuckets=1000, percentiles=[2.0, 98.0] ):

        """
        compute percentile from image histogram - use to rescale 16bit to 8bit
        """

        results = []

        # open image dataset and band 
        src = gdal.Open( pathname )
        for idx in band_idxs: 
            
            # Use GDAL to find the min and max
            band = src.GetRasterBand( idx )
            (lo, hi, avg, std) = band.GetStatistics(True, True)

            # Use GDAL to calculate a big histogram
            rawhist = band.GetHistogram(min=lo, max=hi, buckets=nbuckets)
            binEdges = np.linspace(lo, hi, nbuckets+1)

            # Probability mass function. Trapezoidal-integration of this should yield 1.0.
            pmf = rawhist / (np.sum(rawhist) * np.diff(binEdges[:2]))

            # Cumulative probability distribution. Starts at 0, ends at 1.0.
            distribution = np.cumsum(pmf) * np.diff(binEdges[:2])

            # Which histogram buckets are close to the percentiles requested?
            bucket_idxs = [np.sum(distribution < p / 100.0) for p in percentiles]

            # record result in dict
            results.append (  [binEdges[i] for i in bucket_idxs] )

        return results
