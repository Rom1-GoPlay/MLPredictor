import numpy as np
from matplotlib import pyplot as plt


class MLPredictor:

    X, real_y, m = np.array([[]]), np.array([[]]), 0
    cost_history = []

    def __init__(self):
        print("ml predictor initailaized")

    def load_data(self, data_x, data_y):
        """
        load the given data
        """

        self.n = data_x.shape[1]
        self.theta = np.random.randn(self.n+1, 1)
        self.real_y = data_y
        self.m = data_y.size
        self.x_min = data_x.min()
        self.x_max = data_x.max()
        self.x_mean = np.mean(data_x, axis=0)
        self.x_std = np.std(data_x, axis=0)
        self.X = np.hstack((self.normalize(data_x), np.ones((self.m, 1))))

    def normalize(self, data):
        """
        normalize the given data, then return these normalized data
        """

        return (data - self.x_mean) / self.x_std
        # return (data - self.x_min) / (self.x_max - self.x_min)

    def model(self, inputs=X):
        """
        data must be normalized before beeing passed here
        """

        assert inputs.shape[1] == self.theta.shape[0], f"this shapes are not compatible :\n{inputs.shape} and {self.theta.shape}\n"
        return inputs.dot(self.theta)

    def cost_function(self):
        """
        when model's predictions are accurate this function returns near zero values
        """

        return np.mean((self.model(self.X) - self.real_y)**2) / 2

    def grad(self):
        """
        the cost_function's derivative
        """

        assert self.m != 0, "m=0 so X and real_y are empty"
        return self.X.T.dot(self.model(self.X) - self.real_y) / self.m

    def gradiant_descent_step(self, learning_rate):
        self.theta -= learning_rate * self.grad()
    
    def gradiant_descent(self, learning_rate=0.1, n_iterations=1000):
        """
        that's what makes the model train.
        """

        for iteration_idx in range(n_iterations):
            self.cost_history.append(self.cost_function())
            self.gradiant_descent_step(learning_rate)

    def single_prediction(self, *inputs):
        """
        thanks to this function we can ask the model for a single prediction easily
        no need to add a bias column
        """

        input_arr = np.array(inputs)
        input_arr = self.normalize(input_arr)
        return self.model(np.hstack((input_arr.reshape(1, self.n), np.ones((1, 1)))))
    
    def plot_cost_hist(self):
        """
        this will plot the cost history thanks to pyplot
        """

        plt.plot(self.cost_history)
        plt.show()

    def accuracy(self, x, y):
        """
        This function return something between 0 and 1.
        The more accurate is the model, the closer to 1 this value will be.
        """

        if x.shape[1] == self.theta.shape[0] - 1:
            x = np.hstack((self.normalize(x), np.ones((x.shape[0], 1))))
        elif x.shape[1] != self.theta.shape[0]:
            raise ValueError("incompatible dimensions")

        y_pred = self.model(x)
        u = np.sum((y - y_pred)**2)
        v = np.sum((y - np.mean(y))**2)
        return 1 - u / v



def load_data_from_csv(csv_path, delimiter=";"):
    """
    the y column must be the last
    """
    data = np.genfromtxt(csv_path, delimiter=delimiter, dtype="float")[1:,:]

    data_x = data[:,:-1]
    data_y = data[:,-1:]

    return data_x, data_y



# create an instance of MLPredictor :
my_predictor = MLPredictor()

# load the data inside this instance :
dx, dy = load_data_from_csv("data/generated/three_params_0.csv")
my_predictor.load_data(dx, dy)

# train the model of this instance :
my_predictor.gradiant_descent(n_iterations=500)

# print some informations about the predictor :
print(f"accuracy= {my_predictor.accuracy(dx, dy)}")
print(f"single prediction= {my_predictor.single_prediction([100, 10, 7])}")