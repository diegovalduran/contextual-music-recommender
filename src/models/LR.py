from models.BaseModel import BaseModel
from sklearn.linear_model import LogisticRegression

"""
Logistic Regression classifier implementation.
Extends BaseModel to provide a linear classification model using
scikit-learn's LogisticRegression.
"""
class LR(BaseModel):
	extra_log_args = ['penalty', 'solver']

	@staticmethod
	def parse_model_args(parser):
		"""
		Add Logistic Regression-specific command line arguments.
		
		Args:
			parser: ArgumentParser instance
			
		Returns:
			parser: Updated parser with LR arguments
		"""
		parser.add_argument('--penalty', type=str, default='l2',
						  help='Regularization type (l1, l2, elasticnet, none)')
		parser.add_argument('--regularization', type=float, default=1.0,
						  help='Inverse of regularization strength') 
		parser.add_argument('--max_iter', type=int, default=100,
						  help='Maximum iterations for solver convergence')
		parser.add_argument('--solver', type=str, default='lbfgs',
						  help='Algorithm for optimization') 
		return BaseModel.parse_model_args(parser)

	def __init__(self, args):
		"""
		Initialize LR model with given hyperparameters.
		
		Args:
			args: Parsed command line arguments containing model configuration
				 C = 1/regularization (higher C = less regularization)
		"""
		super().__init__(args)
		self.clf = LogisticRegression(
			random_state=args.random_seed,
			penalty=args.penalty,
			C=1/args.regularization,  # Convert regularization to C parameter
			max_iter=args.max_iter,
			solver=args.solver
		)