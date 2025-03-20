"""
Base class for all machine learning models in the project.
"""
class BaseModel:
	reader = 'BaseReader'
	extra_log_args = []

	@staticmethod
	def parse_model_args(parser):
		"""
		Add model-specific arguments to the argument parser.
		
		Args:
			parser: ArgumentParser instance
			
		Returns:
			parser: Updated ArgumentParser with model arguments
		"""
		parser.add_argument('--model_path', type=str, default='',
							help='Model save path.')
		return parser

	def __init__(self,args):
		"""
		Initialize the base model with common attributes.
		
		Args:
			args: Parsed command line arguments containing model configuration
		"""
		self.model_path = args.model_path
	
	def train(self, X, y):
		"""
		Train the model on our data.
		
		Args:
			X: Feature matrix of shape (n_samples, n_features)
			y: Target values of shape (n_samples,)
			
		Returns:
			self: The trained model instance
		"""
		self.clf = self.clf.fit(X,y)

	def predict(self, X):
		"""
		Make predictions using the trained model.
		
		Args:
			X: Feature matrix of shape (n_samples, n_features)
			
		Returns:
			pred_y: Predicted values for X
		"""
		pred_y = self.clf.predict(X)
		return pred_y
	
	def save(self):
		"""
		Save the model.
		"""
		pass