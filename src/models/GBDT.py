from models.BaseModel import BaseModel
from sklearn.ensemble import GradientBoostingClassifier

"""
Gradient Boosting Decision Tree (GBDT) classifier implementation.
Extends BaseModel for gradient boosting ensemble model using
scikit-learn's GradientBoostingClassifier.
"""
class GBDT(BaseModel):
	extra_log_args = ['lr', 'n_estimators']
	
	@staticmethod
	def parse_model_args(parser):
		"""
		Add GBDT-specific command line arguments.
		
		Args:
			parser: ArgumentParser instance
			
		Returns:
			parser: Updated parser with GBDT arguments
		"""
		parser.add_argument('--lr', type=float, default=0.1,
						  help='Learning rate for boosting')
		parser.add_argument('--n_estimators', type=int, default=100,
						  help='Number of boosting stages')
		parser.add_argument('--subsample', type=float, default=1.0,
						  help='Fraction of samples for stochastic gradient boosting')
		parser.add_argument('--max_depth', type=int, default=None,
						  help='Maximum depth of individual trees')
		parser.add_argument('--min_samples_split', type=int, default=2,
						  help='Minimum samples required to split node')
		parser.add_argument('--min_samples_leaf', type=int, default=1,
						  help='Minimum samples required in leaf node')
		parser.add_argument('--max_leaf_nodes', type=int, default=None,
						  help='Maximum leaf nodes per tree')
		parser.add_argument('--max_features', type=str, default=None,
						  help='Number of features to consider for best split')
		return BaseModel.parse_model_args(parser)

	def __init__(self, args):
		"""
		Initialize GBDT model with given hyperparameters.
		
		Args:
			args: Parsed command line arguments containing model configuration
		"""
		super().__init__(args)
		self.clf = GradientBoostingClassifier(
			random_state=args.random_seed,
			learning_rate=args.lr,
			n_estimators=args.n_estimators,
			subsample=args.subsample,
			max_depth=args.max_depth,
			min_samples_leaf=args.min_samples_leaf,
			min_samples_split=args.min_samples_split,
			max_leaf_nodes=args.max_leaf_nodes,
			max_features=args.max_features,
			validation_fraction=0.125,  # Fraction of training data for early stopping
			n_iter_no_change=10        # Number of iterations with no improvement for early stopping
		)