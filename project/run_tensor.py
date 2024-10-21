"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch

# Use this function to make a random parameter in
# your module.
def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)

# TODO: Implement for Task 2.5.
class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x) -> minitorch.Scalar:
        """
        Forward pass for the network

        Args:
        -----
            x: Tuple of tensors

        Returns:
        -------
            Tensor of forward pass shape
        """
        middle = self.layer1.forward(x).relu_map()
        end = self.layer2.forward(middle).relu_map()
        return self.layer3.forward(end).sigmoid_map()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.in_size = in_size
        self.out_size = out_size

    def forward(self, inputs):
        """Forward pass for the linear layer

        Args:
        -----
            inputs: Tensor of inputs

        Returns:
        -------
            Tensor output
        """

        # def transpose(a: Tensor) -> Tensor:
        #     order = list(range(a.dims))
        #     order[-2], order[-1] = order[-1], order[-2]
        #     return a._new(a._tensor.permute(*order))

        # def conform_shape(data, weights):
        #     data_index = len(data) - 1
        #     weights_index = len(weights) - 1
        #     shape = []
        #     while data_index >= 0 and weights_index >= 0:
        #         if data[data_index] == weights[weights_index]:
        #             shape.append(data[data_index])
        #         elif data[data_index] == 1:
        #             shape.append(weights[weights_index])
        #         elif weights[weights_index] == 1:
        #             shape.append(data[data_index])
        #         else:
        #             shape.append(1)
        #         data_index -= 1
        #         weights_index -= 1

        #     return tuple(shape)

        print(f'input: {inputs.shape} weights: {self.weights.value.shape} bias: {self.bias.value.shape}')
        print(f'---weights: {self.weights.value}')
        print(f'---bias: {self.bias.value}')
        print(f'---inputs: {inputs}')

        broadcast_input_shape = minitorch.shape_broadcast(inputs.shape, self.weights.value.shape)
        print(f'------input_broadcast: {broadcast_input_shape} {inputs.view(*broadcast_input_shape)}')
        out = self.weights.value.backend.mul_zip(self.weights.value, inputs.view(*broadcast_input_shape))

        broadcast_output_shape = minitorch.shape_broadcast(inputs.shape, self.weights.value.shape)
        out = out.backend.add_zip(out, self.bias.value.view(*broadcast_output_shape))
        return input

def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
