import os
import importlib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn + keras
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.image import ImageDataGenerator

# add utility functions
from utility.dp import getUniqueId
from utility.dl.cnn import loadFromFile


def getPrediction( datagen, model, df, data_path ):

	"""
	get regression
	"""

	# create iterator        
	it = datagen.flow_from_dataframe(   dataframe=df,
										directory=data_path,
										x_col='image',
										y_col='target',
										class_mode='raw',
										color_mode='rgb',
										shuffle=False,
										target_size=(128,128),
										batch_size=1 )

	# run prediction
	df[ 'yhat' ] = model.predict_generator( it )
	return df


def plotRegression( dfs ):

	"""
	plot regression
	"""

	# create figure
	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
	for idx, df in enumerate( dfs ):

		# compute regression
		m, c, r2, p, err = stats.linregress( df['target'].values, df['yhat'].values )    

		# plot sample data and regression model
		axes[ idx ].plot( df['target'].values, df['yhat'].values, '.' )
		axes[ idx ].plot( [0, 1], [c, m+c], '-', label='y={:.2f}x+{:.2f}\nR2={:.2f}'.format( m, c, r2 ) )
		axes[ idx ].plot( [0, 1], [0, 1], '--', color='g', label='1-to-1' )

		# fix axes and plot 1-2-1 line
		axes[ idx ].set_xlim([0,1])
		axes[ idx ].set_ylim([0,1])

		subset = 'Train' if idx == 0 else 'Test'
		axes[ idx ].set_title( 'title' )
		axes[ idx ].legend( fontsize=9 )

	plt.show()
	return


def plotHeatMaps( df ):

	"""
	plot heatmaps
	"""

	# create figure
	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

	# cr
	actual  = np.zeros( ( df[ 'y0' ].max() + 128, df[ 'x0' ].max() + 128 ) )
	predict = np.zeros( ( df[ 'y0' ].max() + 128, df[ 'x0' ].max() + 128 ) )

	# iterate rows
	for idx, row in df.iterrows():

		x0 = row[ 'x0' ]
		y0 = row[ 'y0' ]

		# copy actual / predicted target values into heatmap
		actual[ y0:y0+128,x0:x0+128 ] = row[ 'target' ]
		predict[ y0:y0+128,x0:x0+128 ] = row[ 'yhat' ]

	# plot and show
	axes[ 0 ].imshow( actual, vmin=0.1, vmax=0.7 )
	axes[ 1 ].imshow( predict, vmin=0.1, vmax=0.7 )

	plt.show()
	
	return



def parseArguments(args=None):

    """
    parse command line argument
    """

    # parse command line arguments
    parser = argparse.ArgumentParser(description='data prep')
    parser.add_argument('model_path', action="store")
    parser.add_argument('data_path', action="store")

    return parser.parse_args(args)


def main():

	"""
	main path of execution
	"""

	# parse arguments
	args = parseArguments()
	args.image_size = 128

	# load pre-trained model from file 
	model, model_type = loadFromFile( args.model_path )

	# select preprocess_input wrapper
	module = importlib.import_module( 'keras.applications.{}'.format( model_type ) )
	preprocess_input = module.preprocess_input
			
	datagen = ImageDataGenerator(  preprocessing_function=preprocess_input )
	scaler = MinMaxScaler()

	# plot sample size plots and loss diagnostics
	# plotSampleSizes( args.data_path )
	# plotDiagnostics( args.model_path )

	# read dataframe and normalise target
	df_train = pd.read_csv( os.path.join( args.data_path, 'train.csv' ) )
	df_train[ 'target' ] = scaler.fit_transform( df_train[ [ 'target'] ] )

	df_train = getPrediction( datagen, model, df_train, os.path.join( args.data_path, 'train' ) )
	plotHeatMaps( df_train )

	# read dataframe and normalise target
	df_test = pd.read_csv( os.path.join( args.data_path, 'test.csv' ) )
	df_test[ 'target' ] = scaler.transform( df_test[ [ 'target'] ] )

	df_test = getPrediction( datagen, model, df_test, os.path.join( args.data_path, 'test' ) )

	# plot regression
	plotRegression( [ df_train, df_test ] )


	return


# execute main
if __name__ == '__main__':
    main()
