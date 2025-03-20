"""
Main Execution Module for Psychological Music Listening Classification.

This is the primary entry point for training and evaluating
machine learning models on psychological music response data.
"""

import os
import sys
import logging
import argparse
from importlib import reload
import numpy as np
from sklearn.metrics import *
from models import *
from helpers import *
from helpers.configs import *
from helpers import utils

def parse_global_args(parser):
	"""
	Add global arguments to the argument parser.
	
	Args:
		parser (argparse.ArgumentParser): Existing argument parser
		
	Returns:
		parser: Updated argument parser with global arguments
	"""
	parser.add_argument('--verbose', type=int, default=logging.INFO,
					   help="Logging verbosity level")
	parser.add_argument('--log_file', type=str, default='',
					   help="Path to log file")
	parser.add_argument('--random_seed', type=int, default=0,
					   help="Random seed for reproducibility")
	parser.add_argument('--save_prediction', type=int, default=0,
					   help="Whether to save model predictions")
	parser.add_argument('--save_anno', type=str, default='test',
					   help="Annotation for saved files")
	parser.add_argument('--metrics', type=str, default='accuracy,f1,roc_auc,log_loss,rmse,mae',
					   help="Comma-separated list of evaluation metrics")
	parser.add_argument('--class_num', type=int, default=2,
					   help="Number of classes (2 for binary, >2 for multi-class)")
	return parser

def evaluate_binary(pred_y, y, args):
	"""
	Evaluate binary classification predictions using specified metrics.
	
	Args:
		pred_y: Model predictions
		y: True labels
		args: Arguments containing metric specifications
		
	Returns:
		dict: Dictionary of computed metrics
	"""
	select_metrics = args.metrics.split(",")
	results = dict()
	
	# Calculate each requested metric
	for metric in select_metrics:
		if metric in ['accuracy', 'f1', 'roc_auc']:
			# Direct sklearn metrics
			eval_metric = eval(metric+'_score')
		elif metric == 'log_loss':
			# Log loss for predictions
			eval_metric = eval(metric)
		elif metric == 'rmse':
			# Root mean squared error
			eval_metric = eval('mean_squared_error')
			results[metric] = eval_metric(y, pred_y, squared=False)
		elif metric == 'mae':
			# Mean absolute error
			eval_metric = eval('mean_absolute_error')
		else:
			logging.warning("No metric named %s"%(metric))
		
		# Store metric result if not already computed
		if metric not in results:
			results[metric] = eval_metric(y, pred_y)
	return results

def evaluate_multi(pred_y, y, args):
	"""
	Evaluate multi-class classification predictions.
	
	Args:
		pred_y: Model predictions
		y: True labels
		args: Arguments containing metric specifications
		
	Returns:
		dict: Dictionary of computed metrics
	"""
	select_metrics = args.metrics.split(",")
	results = dict()
	
	for metric in select_metrics:
		if metric in ['accuracy']:
			# Standard accuracy
			eval_metric = eval(metric+'_score')
			results[metric] = eval_metric(y, pred_y)
		elif metric == 'macro_f1':
			# Macro-averaged F1 score
			results[metric] = f1_score(y, pred_y, average='macro')
		elif metric == 'micro_f1':
			# Micro-averaged F1 score
			results[metric] = f1_score(y, pred_y, average='micro')
		elif metric in ['macro_ap', 'micro_ap']:
			# Convert to one-hot encoding for AP calculation
			encoder = np.eye(args.class_num)
			y_class = encoder[y.astype(int)]
			y_pred_class = encoder[pred_y.astype(int)] 
			if metric == 'macro_ap':
				results[metric] = average_precision_score(y_class, y_pred_class, average='macro')
			elif metric == 'micro_ap':
				results[metric] = average_precision_score(y_class, y_pred_class, average='micro')
		else:
			logging.warning("No metric named %s"%(metric))
	return results

def evaluate(pred_y, y, args):
	"""
	Route evaluation to appropriate function based on number of classes.
	
	Args:
		pred_y: Model predictions
		y: True labels
		args: Arguments containing evaluation configuration
		
	Returns:
		dict: Computed evaluation metrics
	"""
	if args.class_num == 2:
		results = evaluate_binary(pred_y, y, args)
	else:
		results = evaluate_multi(pred_y, y, args)
	return results

def run():
	"""
	Main training and evaluation pipeline.
	"""
	# 1. Set random seed for reproducibility
	np.random.seed(args.random_seed)
	
	# 2. Load and normalize data
	data = reader_name(args, normalize=True)
	
	# 3. Initialize and train model
	model = model_name(args)
	model.train(data.train_X, data.train_y)

	# 4. Evaluate on all datasets
	for phase, X, y in zip(['train', 'val', 'test'],
						  [data.train_X, data.val_X, data.test_X],
						  [data.train_y, data.val_y, data.test_y]):
		# Get predictions and evaluate
		pred_y = model.predict(X)
		evaluations = evaluate(pred_y, y, args)
		logging.info('%s results -- '%(phase)+utils.format_metric(evaluations))
		
		# Save predictions
		if args.save_prediction:
			np.save(os.path.join(args.save_path, "{}_pred.npy".format(phase)), pred_y)

if __name__=="__main__":
	# Initialize parser with model selection
	init_parser = argparse.ArgumentParser(description='Model')
	init_parser.add_argument('--model_name', type=str, default='LR',
						   help="Name of the model to use")
	init_args, init_extras = init_parser.parse_known_args()
	
	# Dynamically load model and reader classes
	model_name = eval('{0}.{0}'.format(init_args.model_name))
	reader_name = eval('{0}.{0}'.format(model_name.reader))
	
	# Set up complete argument parser
	parser = argparse.ArgumentParser(description='')
	parser = parse_global_args(parser)
	parser = reader_name.parse_data_args(parser)
	parser = model_name.parse_model_args(parser)
	args, extras = parser.parse_known_args()

	# Configure logging and file paths
	log_args = [args.dataname, str(args.random_seed), args.save_anno]
	for arg in model_name.extra_log_args:
		log_args.append(arg + '=' + str(eval('args.' + arg)))
	log_file_name = '__'.join(log_args).replace(' ', '__')
	
	# Adjust paths based on number of classes
	append = "_3class" if args.class_num == 3 else ""
	
	# Set up file paths if not given
	if args.log_file == '':
		args.log_file = '../logs/{}{}/{}/model.txt'.format(init_args.model_name, append, log_file_name)
	if args.model_path == '':
		args.model_path = '../models/{}{}/{}/model.pt'.format(init_args.model_name, append, log_file_name)
	if args.save_prediction:
		args.save_path = "../models/{}{}/{}".format(init_args.model_name, append, log_file_name)
	
	# Create necessary directories
	os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
	os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
	
	# Configure logging
	reload(logging)
	logging.basicConfig(filename=args.log_file, level=args.verbose)
	logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
	
	logging.info("Save model to %s"%(args.model_path))
	logging.info(init_args)

	run()