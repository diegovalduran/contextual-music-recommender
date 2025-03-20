from models.BaseModel import BaseModel
from sklearn.neural_network import MLPClassifier

"""
Multi-Layer Perceptron (MLP) neural network classifier implementation.
Extends BaseModel to provide a deep learning model using scikit-learn's MLPClassifier.
"""
class MLP(BaseModel):
	extra_log_args = []

	@staticmethod
	def parse_model_args(parser):
		"""
		Add MLP-specific command line arguments.
		
		Args:
			parser: ArgumentParser instance
			
		Returns:
			parser: Updated parser with MLP arguments
		"""
		parser.add_argument('--solver', type=str, default='adam',
						  help='Optimization algorithm (adam, sgd, lbfgs)') 
		parser.add_argument('--alpha', type=float, default=1e-4,
						  help='L2 regularization strength')
		parser.add_argument('--batch_size', type=int, default=200,
						  help='Size of minibatches for gradient updates')
		parser.add_argument('--learning_rate', type=float, default=1e-3,
						  help='Initial learning rate')
		parser.add_argument('--hidden_layer_sizes', type=str, default='(100,)',
						  help='Architecture of hidden layers (e.g., (100,) for one layer)')
		parser.add_argument('--activation', type=str, default='relu',
						  help='Activation function (relu, tanh, logistic)') 
		parser.add_argument('--max_iter', type=int, default=1000,
						  help='Maximum number of iterations')
		parser.add_argument('--patience', type=int, default=50,
						  help='Iterations with no improvement before early stopping')
		return BaseModel.parse_model_args(parser)

	def __init__(self, args):
		"""
		Initialize MLP model with given hyperparameters.
		
		Args:
			args: Parsed command line arguments containing model configuration
				 hidden_layer_sizes is evaluated from string to tuple
		"""
		super().__init__(args)
		# Convert string representation of tuple to actual tuple
		hidden_layer_sizes = eval(args.hidden_layer_sizes)
		
		self.clf = MLPClassifier(
			hidden_layer_sizes=hidden_layer_sizes,  
			activation=args.activation,            
			solver=args.solver,                    # Optimization algorithm
			alpha=args.alpha,                      # L2 regularization
			batch_size=args.batch_size,            # Mini-batch size
			learning_rate_init=args.learning_rate, 
			max_iter=args.max_iter,                #
			random_state=args.random_seed,         # For reproducibility
			early_stopping=True,                   # Enable early stopping
			validation_fraction=0.1,               # Fraction of training data for validation
			n_iter_no_change=args.patience         # Early stopping patience
		)
	