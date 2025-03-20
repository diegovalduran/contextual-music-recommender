from models.BaseModel import BaseModel
from sklearn.ensemble import RandomForestClassifier

"""
Random Forest classifier implementation.
Extends BaseModel to provide an ensemble learning model using scikit-learn's
RandomForestClassifier.
"""
class RF(BaseModel):
	extra_log_args = ['n_estimators', 'max_depth', 'max_features']

	@staticmethod
	def parse_model_args(parser):
		"""
		Add Random Forest-specific command line arguments.
		
		Args:
			parser: ArgumentParser instance
			
		Returns:
			parser: Updated parser with RF arguments
		"""
		parser.add_argument('--n_estimators', type=int, default=100,
						  help='Number of trees in the forest')
		parser.add_argument('--max_depth', type=int, default=None,
						  help='Maximum depth of each tree')
		parser.add_argument('--min_samples_split', type=int, default=2,
						  help='Minimum samples required to split node')
		parser.add_argument('--min_samples_leaf', type=int, default=1,
						  help='Minimum samples required in leaf node')
		parser.add_argument('--max_leaf_nodes', type=int, default=None,
						  help='Maximum leaf nodes per tree')
		parser.add_argument('--max_features', type=str, default=None,
						  help='Number of features to consider for best split')
		parser.add_argument('--criterion', type=str, default='gini',
						  help='Split quality measure (gini, entropy)')
		return BaseModel.parse_model_args(parser)

	def __init__(self, args):
		"""
		Initialize Random Forest model with given hyperparameters.
		
		Args:
			args: Parsed command line arguments containing model configuration
		"""
		super().__init__(args)
		self.clf = RandomForestClassifier(
			random_state=args.random_seed,           # For reproducibility
			n_estimators=args.n_estimators,          # Number of trees
			max_depth=args.max_depth,                # Tree depth limit
			min_samples_leaf=args.min_samples_leaf,  # Minimum samples per leaf
			min_samples_split=args.min_samples_split,# Minimum samples for split
			max_leaf_nodes=args.max_leaf_nodes,      # Maximum leaves per tree
			max_features=args.max_features,          # Features to consider per split
			criterion=args.criterion                 # Split quality measure
		)